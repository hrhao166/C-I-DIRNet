import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from functools import lru_cache

import numpy as np

def solve_I_minus_cK(v: torch.Tensor, c: float, eps: float = 1e-14) -> torch.Tensor:
    """
    Efficient version: (I - cK)u = v, solved frequency by frequency with FFT under periodic boundary conditions.
    Keep the solving logic and numerical results consistent with the original implementation (still using float64 by default),
    but vectorize the batch dimension and cache the frequency-domain operator to reduce overhead.

    Parameters:
        v: Tensor with shape (B,1,H,W)
        c: scalar
        eps: Frequency-domain denominator threshold (avoids instability caused by extremely small denominators)

    Returns:
        u: shape (B,1,H,W)
    """
    assert v.ndim == 4 and v.shape[1] == 1, "v must be (B,1,H,W)"
    B, _, H, W = v.shape
    assert H == W, "The current implementation is still consistent with the original version and only demonstrates the square H==W case"

    device = v.device
    # To stay consistent with the original function, float64 is still used here (the original function forced float64)
    work_dtype = torch.float64
    v_work = v.to(work_dtype)

    # -------- Frequency-domain operator cache: \hat{I - cK} --------
    # Note: I_kernel is the unit impulse; K_kernel is the 5-point discrete Laplacian (periodic boundary)
    @lru_cache(maxsize=64)
    def _get_IK_hat(n, c_val, device_str, dtype_str):
        # Construct the kernels only once and run FFT
        I_kernel = torch.zeros((n, n), dtype=work_dtype, device=device)
        I_kernel[0, 0] = 1.0

        K_kernel = torch.zeros((n, n), dtype=work_dtype, device=device)
        K_kernel[0, 0]  = -4.0
        K_kernel[0, 1]  =  1.0
        K_kernel[0, -1] =  1.0
        K_kernel[1, 0]  =  1.0
        K_kernel[-1, 0] =  1.0

        I_hat = torch.fft.fft2(I_kernel)
        K_hat = torch.fft.fft2(K_kernel)
        IK_hat = I_hat - c_val * K_hat
        return IK_hat

    IK_hat = _get_IK_hat(H, float(c), str(device), str(work_dtype))

    # -------- Batch into the frequency domain and solve all at once for each frequency --------
    # v[:,0,:,:] -> (B,H,W) complex spectrum
    v_hat = torch.fft.fft2(v_work[:, 0, :, :])

    denom = IK_hat  # (H,W), complex
    # Consistent with the original logic: frequency points with denominators that are too small are set to 0
    safe_mask = denom.abs() > eps
    u_hat = torch.zeros_like(v_hat, dtype=torch.complex128)
    # Only divide at safe positions (broadcast to batch)
    u_hat[:, safe_mask] = v_hat[:, safe_mask] / denom[safe_mask]

    # Inverse FFT back to the spatiotemporal domain and take the real part
    u_space = torch.fft.ifft2(u_hat).real  # (B,H,W)

    # Restore to (B,1,H,W) and keep dtype (float64) consistent with the original function
    u_out = u_space.unsqueeze(1).to(work_dtype)
    return u_out


def LaplacianKernel(requires_grad: bool = False):
    """
    Return a 3x3 discrete Laplacian convolution kernel with shape (1,1,3,3).
    The values are [[0,1,0],[1,-4,1],[0,1,0]], corresponding to ∇^2 f.
    """
    # Define the 3x3 kernel
    kernel = torch.tensor(
        [[0., 1., 0.],
         [1., -4., 1.],
         [0., 1., 0.]]
    )
    # Adjust the shape to (out_channels=1, in_channels=1, H=3, W=3)
    kernel = kernel.view(1, 1, 3, 3)
    kernel.requires_grad_(requires_grad)
    return kernel



class POTTS(nn.Module):
    def __init__(self,args):
        
        super(POTTS,self).__init__()
        # self.sig=nn.Sigmoid()
        # self.relu=nn.ReLU(inplace=True)

        self.tau=args.tau # Time step, Delta t=0.5
        self.Theta=args.Theta # Weight of the Cauchy-Riemann constraint
        # self.iter_num=max(abs(args.iter_num),1) # Number of iterations for each point, set to 1 here
        self.device=args.device

        self.mid_channels=args.mid_channels # Corresponds to the number of channels at each grid level j in parallel splitting, namely {c_1,c_2,c_3,c_4,c_5}, set to {32,32,64,128,256}
        self.level_max=len(args.mid_channels) # Maximum grid level, i.e. 5
        self.kernel_size_max=3**self.level_max # Maximum convolution kernel size, i.e. 243
        self.skip_connection=args.connect
        self.times_list=args.times_list # Corresponds to the number of convolution layers at each grid level j in sequential splitting, namely {L_1,L_2,L_3,L_4,L_5}, set to {3,3,3,5,5}
        self.num_blocks=args.num_blocks # Number of time steps, set to 4, i.e. U^1->U^2->U^3->U^4, corresponding to N=4 in the paper
        self.cascade_nums=args.cascade_nums # Number of cascades, set to 3
        self.kernel_size_bound=args.kernel_size_bound # Upper bound of the convolution kernel size, i.e. 5
        self.BNLearn=args.BNLearn
        
        self.tau_explicit=args.tau_explicit
        # self.lambdaLearn=args.lambdaLearn
        # self.bLearn=args.bLearn

        # self.LaplaceAct=LaplaceAct1(args=args)
    
    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2)
                )
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2)
                )
        return layer

class LaplaceAct1(POTTS):
# Activation function that ensures the deformation field satisfies the Cauchy-Riemann constraint
    def __init__(self,args=None):
        super().__init__(args=args)
        self.c=self.Theta*self.tau

    def forward(self,v):

        u=solve_I_minus_cK(v,self.c)
        return u

# class LaplaceAct1(POTTS):
# # Activation function that ensures the deformation field satisfies the Cauchy-Riemann constraint
#     def __init__(self,args=None):
#         super().__init__(args=args)
#         self.c=self.Theta*self.tau
#         self.LaplaceKernel=LaplacianKernel().to(self.device)

#     def forward(self,v):
#         u=v+self.c*F.conv2d(v,self.LaplaceKernel,padding=1)
#         return u

class MGPCConv_first(POTTS):
# finest grid level on the left branch of the V-cycle
    def __init__(self, args=None, times=2, in_channels=2,out_channels=32,kernel_size=None): # in_channels and out_channels refer to the input and output channels of a single convolution kernel convolved with u1 or u2
        super().__init__(args=args) #in_channels=1, out_channels=32, kernel_size=5

        self.in_channels=in_channels # in_channels = 1
        self.out_channels=out_channels # out_channels = mid_channels[0] = 32, i.e. c_1=32
        self.times=times # times = times_list[0] = 3, i.e. L_1=3
        self.kernel_size=np.minimum(kernel_size,self.kernel_size_bound)
        # kernel_size_max = 243, kernel_size_bound = 5, kernel_size = 5, seems inconsistent with the paper; the convolution kernel size of the first layer in the left branch is 3 in the paper 

        self.model=nn.ModuleList() # Start building the network
        self.model.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,
                      stride=1,padding=1,bias=False)) # Used to receive u1,u2
        self.model.append(nn.BatchNorm2d(self.out_channels,affine=self.BNLearn)) # out_channels = 32, normalization layer for u1,u2; at this point, the l=1 sequential splitting layer is built
        self.model.append(nn.LeakyReLU(0.2))

        for i in np.arange(self.times-1): # times = 3; subtract 1 here because the first layer has already been added, so only 2 more layers need to be added, i = 0,1, corresponding to the two sequential splitting layers l=2,3
            self.model.append(nn.Conv2d(in_channels=self.out_channels,out_channels=self.out_channels, 
                      kernel_size=self.kernel_size,
                      stride=1,padding=1,bias=False)) # Used to receive the out_channels components of processed u1,u2
            self.model.append(nn.BatchNorm2d(self.out_channels,affine=self.BNLearn))
            self.model.append(nn.LeakyReLU(0.2))
            '''
            in_channels = out_channels = 32,Here the number of in_channels equals the output channel count of the level l-1 sequential splitting network, equivalent to c_{j,l}, 
            out_channels = 32, also equals the number of in_channels of the level l+1 sequential splitting network, kernel_size = 5
            At this point, the l=2,3 sequential splitting layers are built
            '''
            # self.model.append(self.sigAct1)

    def forward(self,x,src, tgt): # flist is not needed in the registration task
        T_x1, T_x2 = torch.gradient(src, dim=(2, 3)) # Compute the gradient of src; T_x1 is the gradient in the x1 direction, and T_x2 is the gradient in the x2 direction
        x=x.float()
        C = x.shape[1]  # Number of channels
        x_ch1 = x[:, :C//2, :, :]  # First half of channels
        x_ch2 = x[:, C//2:, :, :]  # Second half of channels
        
        # x_ch1=x_ch1.float()
        # x_ch2=x_ch2.float()
        if self.tau_explicit:
            out=self.out_channels*self.tau*self.model[0](x) # +torch.sum(x_ch1,dim=1,keepdim=True)/self.in_channels-self.out_channels*self.tau*(src-tgt)*T_x1
            out1 = out[:, :self.out_channels//2, :, :]       
            out2 = out[:, self.out_channels//2:, :, :]                                           # +torch.sum(x_ch2,dim=1,keepdim=True)/self.in_channels-self.out_channels*self.tau*(src-tgt)*T_x2
            out1=out1+torch.sum(x_ch1,dim=1,keepdim=True)/self.in_channels-(self.out_channels*self.tau*(src-tgt)*T_x1)/self.in_channels
            out2=out2+torch.sum(out1,dim=1,keepdim=True)/self.in_channels-(self.out_channels*self.tau*(src-tgt)*T_x2)/self.in_channels
            out=torch.cat([out1,out2],dim=1)
            # Solve equation (4.25) in the l=1 sequential splitting layer
        else:
            out=self.model[0](x)
        out=self.model[1](out)
        out=self.model[2](out)

        for i in np.arange(self.times-1): # times = 3; subtract 1 here because the first layer has already been passed, so only 2 more layers need to be passed, i = 0,1
            # out1_temp=out1
            # out2_temp=out2
            if self.tau_explicit:
                out1_temp=out[:, :self.out_channels//2, :, :]
                out2_temp=out[:, self.out_channels//2:, :, :]
                out=self.out_channels*self.tau*self.model[i*3+3](out)
                out1 = out[:, :self.out_channels//2, :, :]
                out2 = out[:, self.out_channels//2:, :, :]
                out1=out1+torch.sum(out1_temp,dim=1,keepdim=True)/self.out_channels-(self.out_channels*self.tau*(src-tgt)*T_x1)/self.out_channels
                out2=out2+torch.sum(out2_temp,dim=1,keepdim=True)/self.out_channels-(self.out_channels*self.tau*(src-tgt)*T_x2)/self.out_channels
                out=torch.cat([out1,out2],dim=1)
            else:
                out=self.model[i*3+3](out)
                
            # Solve equation (4.25) in the l=2,3 sequential splitting layers
            out=self.model[i*3+4](out)

        
        return out 
    

class MGPCConv_down(POTTS):
# the rest of the grid levels of the left branch
    def __init__(self,args=None, times=2, in_channels=32,out_channels=32,output_level=1): # in_channels and out_channels refer to the input and output channels of a single convolution kernel convolved with u1 or u2
        super().__init__(args=args)

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.output_level=output_level
        self.times=times
        self.kernel_size=np.minimum(self.kernel_size_max//(3**output_level),self.kernel_size_bound)
        # self.pooling=nn.MaxPool2d(kernel_size=2,stride=2)

        self.model=nn.ModuleList()
        self.model.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,
                      stride=1,padding=1,bias=False)) # Used to receive u1,u2
        self.model.append(nn.BatchNorm2d(out_channels,affine=self.BNLearn)) # out_channels = 32, normalization layer for u1,u2; at this point, the l=1 sequential splitting layer is built
        self.model.append(nn.LeakyReLU(0.2))

        for i in np.arange(self.times-1): # times = 3; subtract 1 here because the first layer has already been added, so only 2 more layers need to be added, i = 0,1, corresponding to the two sequential splitting layers l=2,3
            self.model.append(nn.Conv2d(in_channels=self.out_channels,out_channels=self.out_channels, 
                      kernel_size=self.kernel_size,
                      stride=1,padding=1,bias=False)) # Used to receive the out_channels components of processed u1,u2
            self.model.append(nn.BatchNorm2d(self.out_channels,affine=self.BNLearn))
            self.model.append(nn.LeakyReLU(0.2))
            '''
            in_channels = out_channels = 32,Here the number of in_channels equals the output channel count of the level l-1 sequential splitting network, equivalent to c_{j,l}, 
            out_channels = 32, also equals the number of in_channels of the level l+1 sequential splitting network, kernel_size = 5
            At this point, the l=2,3 sequential splitting layers are built
            '''
            
    def forward(self,x,src,tgt): # flist is not needed in the registration task
        T_x1, T_x2 = torch.gradient(src, dim=(2, 3)) # Compute the gradient of src; T_x1 is the gradient in the x1 direction, and T_x2 is the gradient in the x2 direction
        x=x.float()
        C = x.shape[1]  # Number of channels
        x_ch1 = x[:, :C//2, :, :]  # First half of channels
        x_ch2 = x[:, C//2:, :, :]  # Second half of channels
        if self.tau_explicit:
            out=self.out_channels*self.tau*self.model[0](x)
            out1 = out[:, :self.out_channels//2, :, :]
            out2 = out[:, self.out_channels//2:, :, :]
            out1=out1+torch.sum(x_ch1,dim=1,keepdim=True)/self.in_channels-(self.out_channels*self.tau*(src-tgt)*T_x1)/self.in_channels
            out2=out2+torch.sum(out1,dim=1,keepdim=True)/self.in_channels-(self.out_channels*self.tau*(src-tgt)*T_x2)/self.in_channels
            out=torch.cat([out1,out2],dim=1)
            
            # Solve equation (4.25) in the l=1 sequential splitting layer
        else:
            out=self.model[0](x)
        out=self.model[1](out)
        out=self.model[2](out)

        for i in np.arange(self.times-1): # times = 3; subtract 1 here because the first layer has already been passed, so only 2 more layers need to be passed, i = 0,1
            # out1_temp=out1
            # out2_temp=out2
            if self.tau_explicit:
                out1_temp=out[:, :self.out_channels//2, :, :]
                out2_temp=out[:, self.out_channels//2:, :, :]
                out=self.out_channels*self.tau*self.model[i*3+3](out)
                out1 = out[:, :self.out_channels//2, :, :]
                out2 = out[:, self.out_channels//2:, :, :]
                out1=out1+torch.sum(out1_temp,dim=1,keepdim=True)/self.out_channels-(self.out_channels*self.tau*(src-tgt)*T_x1)/self.out_channels
                out2=out2+torch.sum(out2_temp,dim=1,keepdim=True)/self.out_channels-(self.out_channels*self.tau*(src-tgt)*T_x2)/self.out_channels
                out=torch.cat([out1,out2],dim=1)
            else:
                out=self.model[i*3+3](out)
            # Solve equation (4.25) in the l=2,3 sequential splitting layers
            out=self.model[i*3+4](out)
            out=self.model[i*3+5](out)

        return out

    


class MGPCConv_up(POTTS):
# right branch of the V-cycle
    def __init__(self, args=None, times=2, in_channels=32,out_channels=64,output_level=1):
        super().__init__(args=args)

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.output_level=output_level
        self.times=times
        self.kernel_size=np.minimum(self.kernel_size_max//(3**output_level),self.kernel_size_bound)

        self.model=nn.ModuleList()
        self.model.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,
                      stride=1,padding=1,bias=False)) # Used to receive u1,u2
        self.model.append(nn.BatchNorm2d(self.out_channels,affine=self.BNLearn)) # out_channels = 32, normalization layer for u1,u2; at this point, the l=1 sequential splitting layer is built
        self.model.append(nn.LeakyReLU(0.2))

        for i in np.arange(self.times-1): # times = 3; subtract 1 here because the first layer has already been added, so only 2 more layers need to be added, i = 0,1, corresponding to the two sequential splitting layers l=2,3
            self.model.append(nn.Conv2d(in_channels=self.out_channels,out_channels=self.out_channels, 
                      kernel_size=self.kernel_size,
                      stride=1,padding=1,bias=False)) # Used to receive the out_channels components of processed u1
            self.model.append(nn.BatchNorm2d(self.out_channels,affine=self.BNLearn))
            self.model.append(nn.LeakyReLU(0.2))
            '''
            in_channels = out_channels = 32,Here the number of in_channels equals the output channel count of the level l-1 sequential splitting network, equivalent to c_{j,l}, 
            out_channels = 32, also equals the number of in_channels of the level l+1 sequential splitting network, kernel_size = 5
            At this point, the l=2,3 sequential splitting layers are built
            '''
            
    def forward(self,x,src,tgt): # flist is not needed in the registration task
        T_x1, T_x2 = torch.gradient(src, dim=(2, 3)) # Compute the gradient of src; T_x1 is the gradient in the x1 direction, and T_x2 is the gradient in the x2 direction
        C = x.shape[1]  # Number of channels
        x_ch1_1 = x[:, :C//4, :, :]  # First half of channels
        x_ch2_1 = x[:, C//4:C//2, :, :]  # Second half of channels
        x_ch1_2 = x[:, C//2:3*C//4, :, :]  # First half of channels
        x_ch2_2 = x[:, 3*C//4:, :, :]  # Second half of channels
        x_ch1=torch.cat([x_ch1_1,x_ch1_2],dim=1) # (B,out_channels,H,W), u1 part
        x_ch2=torch.cat([x_ch2_1,x_ch2_2],dim=1) # (B,out_channels,H,W), u2 part
        if self.tau_explicit:
            out=self.out_channels*self.tau*self.model[0](x)
            out1 = out[:, :self.out_channels//2, :, :]
            out2 = out[:, self.out_channels//2:, :, :]
            out1=out1+torch.sum(x_ch1,dim=1,keepdim=True)/self.in_channels-(self.out_channels*self.tau*(src-tgt)*T_x1)/self.in_channels
            out2=out2+torch.sum(out1,dim=1,keepdim=True)/self.in_channels-(self.out_channels*self.tau*(src-tgt)*T_x2)/self.in_channels
            out=torch.cat([out1,out2],dim=1)

            # Solve equation (4.25) in the l=1 sequential splitting layer
        else:
            out=self.model[0](x)
        out=self.model[1](out)
        out=self.model[2](out)

        for i in np.arange(self.times-1): # times = 3; subtract 1 here because the first layer has already been passed, so only 2 more layers need to be passed, i = 0,1
            # out1_temp=out1
            # out2_temp=out2
            if self.tau_explicit:
                out1_temp=out[:, :self.out_channels//2, :, :]
                out2_temp=out[:, self.out_channels//2:, :, :]
                out=self.out_channels*self.tau*self.model[i*3+3](out)
                out1 = out[:, :self.out_channels//2, :, :]
                out2 = out[:, self.out_channels//2:, :, :]
                out1=out1+torch.sum(out1_temp,dim=1,keepdim=True)/self.out_channels-(self.out_channels*self.tau*(src-tgt)*T_x1)/self.out_channels
                out2=out2+torch.sum(out2_temp,dim=1,keepdim=True)/self.out_channels-(self.out_channels*self.tau*(src-tgt)*T_x2)/self.out_channels
                out=torch.cat([out1,out2],dim=1)
            else:
                out=self.model[i*3+3](out)
            # Solve equation (4.25) in the l=2,3 sequential splitting layers
            out=self.model[i*3+4](out)
            out=self.model[i*3+5](out)


        return out

class Block(POTTS):
# construct blocks. Each block is a time step. 
    def __init__(self,args=None,in_channels=1):
        super(Block,self).__init__(args=args)
        self.LaplaceAct=LaplaceAct1(args=args)

        self.downs=nn.ModuleList()
        self.downs.append(MGPCConv_first(args=args, times=self.times_list[0], in_channels=2, 
                        out_channels=self.mid_channels[0], kernel_size=self.kernel_size_max,))
        self.pooling=nn.MaxPool2d(kernel_size=2,stride=2) # Downsample
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') # Upsample 
        # out_channels = mid_channels[0] = 32, times = times_list[0]=3, kernel_size = kernel_size_max=243
        # self.times_list = {L_1,L_2,L_3,L_4,L_5} = [3,3,3,5,5], self.mid_channels = {c_1,c_2,c_3,c_4,c_5} = [32,32,64,128,256]
        for i in np.arange(self.level_max-1): # level_max = 5, i = 0,1,2,3, subtract 1 because the first grid level has already been added
            self.downs.append(MGPCConv_down(args=args, times=self.times_list[i+1], in_channels=self.mid_channels[i],
                                             out_channels=self.mid_channels[i+1],output_level=i+1,))
            # The in_channels of the next grid level equals the out_channels of the previous grid level
        
        self.ups=nn.ModuleList()    
        # self.combineWeight=nn.Parameter(torch.randn(self.level_max-1,2), requires_grad=True)
        for i in np.flip(np.arange(self.level_max-1)): # level_max = 5, i = 3,2,1,0,subtract 1 because the first grid level has already been added
            self.ups.append(MGPCConv_up(args=args, times=self.times_list[i], in_channels=self.mid_channels[i] + self.mid_channels[i+1],
                                             out_channels=self.mid_channels[i],output_level=i,))
            
        self.final=nn.Conv2d(in_channels=self.mid_channels[0]+2,out_channels=2,kernel_size=3, stride=1, padding=1,
                      bias=True) # Used to receive u1
        
        

    def forward(self,x,src, tgt): # flist is not needed in the registration task
        out=x
        x=x.float()
        # x_enc = [x] # Add x to the x_enc list
        connect=[]
        for i in np.arange(self.level_max): # level_max = 5, i = 0,1,2,3,4

            out=self.downs[i](out,src,tgt) # out is the output of grid level j at index i in the left branch
            if i<self.level_max-1: # level_max-1 = 4, i = 0,1,2,3
                # x_enc.append(out) # Add the output out of each grid level j in the left branch to the x_enc list
                connect.append(out) # Add the output out of each grid level j in the left branch to the connect list
                out=self.pooling(out) # Downsample
                src = self.pooling(src) # Downsample
                tgt = self.pooling(tgt) # Downsample
            

        # out is the output of the last grid level j in the left branch
        connect=connect[::-1]

        for i in np.arange(self.level_max-1): # level_max-1 = 4, i = 0,1,2,3
            out=self.upsample(out) # Upsample
            src=self.upsample(src) # Upsample
            tgt=self.upsample(tgt) # Upsample
            

            out = torch.cat([out, connect[i]], dim=1)
            out=self.ups[i](out,src,tgt) # out is the output of grid level j at index i in the right branch

        if self.tau_explicit:
            out = torch.cat([out,x], dim=1) # out is the output of grid level j at index i in the right branch
            C = out.shape[1]  # Number of channels
            # C2 = x.shape[1]  # Number of channels
            
            out1_temp = out[:, :C//2, :, :]
            out2_temp = out[:, C//2:, :, :]
            # out=torch.cat([out,x],dim=1) # out is the output of grid level j at index i in the right branch
            out=self.final(out) # out is the output of the deformation field
            out1 = out[:, 0:1, :, :]
            out2 = out[:, 1:2, :, :]
            out1=out1+torch.sum(out1_temp,dim=1,keepdim=True)/self.mid_channels[0]
            out2=out2+torch.sum(out2_temp,dim=1,keepdim=True)/self.mid_channels[0]
            # Solve equation (4.25) in the l=1 sequential splitting layer
        else:
            out=self.final(out)
        
        out1=self.LaplaceAct(out1)
        out2=self.LaplaceAct(out2)

        out=torch.cat([out1,out2],dim=1)


        return out
    
class layer1(POTTS):
# layer1 preprocesses the input image pair to initialize the deformation field
    def __init__(self, args=None, in_channels=1):
        super(layer1,self).__init__(args=args)
        self.dim=args.dim
        self.bn=args.bn
        
        
        self.layer1=nn.ModuleList()
        self.layer1.append(self.conv_block(dim=self.dim, in_channels=2, out_channels=2, batchnorm=self.bn))
        # layer1 is responsible for preprocessing the input image pair to initialize the deformation field
    
    def forward(self,x):
        x = x.float()
        out=self.layer1[0](x)
        return out # out is the initial value of the deformation field


class POTTSNET(POTTS):
# assemble blocks
    def __init__(self, args=None, in_channels=1):
        super(POTTSNET,self).__init__(args=args)
        self.dim=args.dim
        self.bn=args.bn
        
        self.layer1=nn.ModuleList()
        self.layer1=layer1(args=args) # layer1 preprocesses the input image pair to initialize the deformation field
        

        self.blocks = nn.ModuleList() # Used to store each layer of the encoder
        for i in np.arange(self.num_blocks): # num_blocks = 4, i = 0,1,2,3
            self.blocks.append(Block(args=args))

    def forward(self, src, tgt):

        x = torch.cat([src, tgt], dim=1) # Concatenate the moving image and target image; dim=1 means concatenation along the channel dimension
        
        out=self.layer1(x) # x is the original image pair; layer1 preprocesses it, and out is the initial value of the deformation field
        for idx in range(self.num_blocks):
            out=self.blocks[idx](out,src,tgt) # out is the output of the deformation field
        
        out[:, :, 0, :] = 0
        # bottom row
        out[:, :, -1, :] = 0
        # left column
        out[:, :, :, 0] = 0
        # right column
        out[:, :, :, -1] = 0

        return out # out is the final value of the deformation field
    

# class SpatialTransformer(nn.Module):
#     def __init__(self, size, mode='bilinear'):
#         super(SpatialTransformer, self).__init__()
#         # Create sampling grid
#         vectors = [torch.arange(0, s) for s in size]
#         grids = torch.meshgrid(vectors)
        
#         grid = torch.stack(grids)  # y, x, z
        
#         grid = torch.unsqueeze(grid, 0)  # add batch
#         grid = grid.type(torch.FloatTensor)
#         self.register_buffer('grid', grid)

#         self.mode = mode

#     def forward(self, src, flow):
#         src=src.float()
#         new_locs = self.grid + flow
#         # print('new_locs:',new_locs)
        
#         shape = flow.shape[2:]

#         # Need to normalize grid values to [-1, 1] for resampler
#         for i in range(len(shape)):
#             new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        
        
#         if len(shape) == 2:
#             new_locs = new_locs.permute(0, 2, 3, 1)
#             new_locs = new_locs[..., [1, 0]]
#             new_locs = new_locs.float()
#         elif len(shape) == 3:
#             new_locs = new_locs.permute(0, 2, 3, 4, 1)
#             new_locs = new_locs[..., [2, 1, 0]]
#             new_locs = new_locs.float()

#         return F.grid_sample(src, new_locs, mode=self.mode)

class SpatialTransformer(nn.Module):
    """
    Spatial transformer using direct coordinate stacks (second method):
    - For 2D/3D input, dynamically generate base coordinates.
    - Add the displacement field (flow) predicted by the network to the base coordinates to obtain sampling positions.
    - After normalizing the sampling positions to [-1,1], call grid_sample to complete interpolation transformation.
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.size = size  # 2D: (H, W), 3D: (D, H, W)
        self.mode = mode

    def forward(self, src, flow):
        # src: (B, C, ... spatial dims ...)
        # flow: (B, ndim, ... spatial dims ...)
        src=src.float()
        B = src.shape[0]
        device = flow.device
        dims = len(self.size)

        if dims == 2:
            H, W = self.size
            # Build base grid coordinates from 1 to W/H
            x = torch.linspace(1, W, W, device=device)
            y = torch.linspace(1, H, H, device=device)
            x_coords = x.unsqueeze(0).unsqueeze(1).repeat(B, H, 1)  # (B, H, W)
            y_coords = y.unsqueeze(0).unsqueeze(2).repeat(B, 1, W)  # (B, H, W)
            base_grid = torch.stack((x_coords, y_coords), dim=-1)   # (B, H, W, 2)

            # Add displacement
            disp = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
            total_grid = base_grid + disp

            # Normalize to [-1,1]: x_norm = (x - (W+1)/2) / ((W-1)/2)
            total_grid[..., 0] = (total_grid[..., 0] - (W+1)/2) / ((W-1)/2)
            total_grid[..., 1] = (total_grid[..., 1] - (H+1)/2) / ((H-1)/2)

        else:
            D, H, W = self.size
            # 3D coordinate generation
            z = torch.linspace(1, D, D, device=device)
            y = torch.linspace(1, H, H, device=device)
            x = torch.linspace(1, W, W, device=device)
            zc = z.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B, D, H, W)
            yc = y.unsqueeze(0).unsqueeze(2).unsqueeze(1).repeat(B, D, H, W)
            xc = x.unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(B, D, H, W)
            base_grid = torch.stack((xc, yc, zc), dim=-1)  # (B, D, H, W, 3)

            disp = flow.permute(0, 2, 3, 4, 1)             # (B, D, H, W, 3)
            total_grid = base_grid + disp

            total_grid[..., 0] = (total_grid[..., 0] - (W+1)/2) / ((W-1)/2)
            total_grid[..., 1] = (total_grid[..., 1] - (H+1)/2) / ((H-1)/2)
            total_grid[..., 2] = (total_grid[..., 2] - (D+1)/2) / ((D-1)/2)

        # Call grid_sample to complete the spatial transformation
        total_grid = total_grid.float()
        return F.grid_sample(
            src,
            total_grid,
            mode=self.mode,
            align_corners=True
        )




