import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np

def solve_I_minus_cK(v, c):# 这个函数被用于在最细的网格输出时，保证形变场满足柯西-黎曼约束
    """
    Use FFT to solve (I - cK)u = v under periodic boundary condition (circular convolution).
    Extended to handle batch input of shape (B, 1, H, W).

    :param v: input tensor of shape (B, 1, H, W)
              B: batch size, 1: single channel, H,W: image/field size
    :param c: scalar constant
    :return: u of shape (B, 1, H, W), the solution for each batch
    """

    # 检查输入形状：v.shape => (B, C, H, W)
    B, C, H, W = v.shape
    assert C == 1, "This function expects a single-channel input, shape=(B,1,H,W)."
    assert H == W, "Example code only demonstrates square shape H==W."

    # ---------- 准备 (I - cK) 核的 FFT ----------
    # 只需根据 H,W 构造一次 I_kernel, K_kernel
    n = H
    # I_kernel: 单位脉冲 (0,0)=1，其余=0
    
    I_kernel = torch.zeros((n, n), dtype=torch.float64)
    I_kernel[0, 0] = 1.0

    # K_kernel: 离散拉普拉斯 (周期边界)
    
    K_kernel = torch.zeros((n, n), dtype=torch.float64)
    K_kernel[0, 0] = -4.0
    K_kernel[0, 1]  = 1.0   # right
    K_kernel[0, -1] = 1.0   # left
    K_kernel[1, 0]  = 1.0   # down
    K_kernel[-1, 0] = 1.0   # up

    
    I_hat = torch.fft.fft2(I_kernel)
    K_hat = torch.fft.fft2(K_kernel)
    IK_hat = I_hat - c * K_hat   # (I - cK) in freq domain

    # 为输出分配空间 (B,1,H,W)
    u_out = torch.zeros_like(v, dtype=torch.float64)  # 或者和 v.dtype 一致

    eps = 1e-14

    # ---------- 对批次中每个样本逐一处理 ----------
    for b in range(B):
        # 取出当前样本 (1,H,W) -> (H,W)
        v_slice = v[b, 0, :, :]

        # 转为复数做 FFT
        
        v_hat = torch.fft.fft2(v_slice)

        # 计算 u_hat = v_hat / (I - cK), 若分母太小则置 0
        u_hat = torch.zeros_like(v_hat, dtype=torch.complex128)

        for i in range(n):
            for j in range(n):
                denom = IK_hat[i, j]
                if abs(denom) > eps:
                    u_hat[i, j] = v_hat[i, j] / denom
                else:
                    # 分母≈0，通常是 DC 分量或与核的零模相关
                    # 这里简单置0。更复杂场景可做正则处理
                    u_hat[i, j] = 0.0

        # iFFT 得到空间域结果

        u_slice = torch.fft.ifft2(u_hat).real  # shape (H,W)

        # 放回 batch 输出
        u_out[b, 0, :, :] = u_slice

    return u_out


class POTTS(nn.Module):
    def __init__(self,args):
        
        super(POTTS,self).__init__()
        # self.sig=nn.Sigmoid()
        # self.relu=nn.ReLU(inplace=True)

        self.tau=args.tau # 时间步长，△t=0.5
        self.Theta=args.Theta # 柯西-黎曼约束的权重
        # self.iter_num=max(abs(args.iter_num),1) # 每个点的迭代次数，这里设置为1
        self.device=args.device

        self.mid_channels=args.mid_channels # 对应平行分割中每一个网格级别j的通道数，即{c_1,c_2,c_3,c_4,c_5}，设置为{32,32,64,128,256}
        self.level_max=len(args.mid_channels) # 网格级别的最大值，即5
        self.kernel_size_max=3**self.level_max # 最大的卷积核大小，即243
        self.skip_connection=args.connect
        self.times_list=args.times_list # 对应顺序分割的每一个网格级别j的卷积层数，即{L_1,L_2,L_3,L_4,L_5}，设置为{3,3,3,5,5}
        self.num_blocks=args.num_blocks # 时间步数，设置为4，即U^1->U^2->U^3->U^4，对应着论文中的N=4
        self.cascade_nums=args.cascade_nums # 级联的次数，设置为3
        self.kernel_size_bound=args.kernel_size_bound # 卷积核的尺寸上限，即5
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
# 保证形变场满足柯西黎曼约束的激活函数
    def __init__(self,args=None):
        super().__init__(args=args)
        self.c=self.Theta*self.tau

    def forward(self,v):

        u=solve_I_minus_cK(v,self.c)
        return u

class MGPCConv_first(POTTS):
# finest grid level on the left branch of the V-cycle
    def __init__(self, args=None, times=2, in_channels=2,out_channels=32,kernel_size=None): # in_channels和out_channels指的是与u1或u2做卷积的单个卷积核的输入和输出通道数
        super().__init__(args=args) #in_channels=1, out_channels=32, kernel_size=5

        self.in_channels=in_channels # in_channels = 1
        self.out_channels=out_channels # out_channels = mid_channels[0] = 32, 即c_1=32
        self.times=times # times = times_list[0] = 3, 即L_1=3
        self.kernel_size=np.minimum(kernel_size,self.kernel_size_bound)
        # kernel_size_max = 243, kernel_size_bound = 5, kernel_size = 5, 与论文似乎不一致，论文中左分支的第一层的卷积核大小是3 

        self.model=nn.ModuleList() # 开始构建网络
        self.model.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,
                      stride=1,padding=1,bias=False)) # 用于接收u1,u2
        self.model.append(nn.BatchNorm2d(self.out_channels,affine=self.BNLearn)) # out_channels = 32，u1,u2的归一化层，至此，l=1的顺序分割层搭建完成
        self.model.append(nn.LeakyReLU(0.2))

        for i in np.arange(self.times-1): # times = 3，这里减1是因为第一层已经添加了，所以只需要再添加2层, i = 0,1, 对应l=2,3两个顺序分割层
            self.model.append(nn.Conv2d(in_channels=self.out_channels,out_channels=self.out_channels, 
                      kernel_size=self.kernel_size,
                      stride=1,padding=1,bias=False)) # 用于接收处理过后的u1,u2的out_channels个分量
            self.model.append(nn.BatchNorm2d(self.out_channels,affine=self.BNLearn))
            self.model.append(nn.LeakyReLU(0.2))
            '''
            in_channels = out_channels = 32,这里的in_channels数量等于l-1级顺序分割网络的输出通道数, 相当于c_{j,l}, 
            out_channels = 32, 也等于l+1级顺序分割网络的in_channels数量, kernel_size = 5
            至此，l=2,3的顺序分割层搭建完成
            '''
            # self.model.append(self.sigAct1)

    def forward(self,x,src, tgt): # 在配准任务中不需要flist
        T_x1, T_x2 = torch.gradient(src, dim=(2, 3)) # 计算src的梯度，T_x1是x1方向的梯度，T_x2是x2方向的梯度
        x=x.float()
        C = x.shape[1]  # 通道数
        x_ch1 = x[:, :C//2, :, :]  # 前一半通道
        x_ch2 = x[:, C//2:, :, :]  # 后一半通道
        
        # x_ch1=x_ch1.float()
        # x_ch2=x_ch2.float()
        if self.tau_explicit:
            out=self.out_channels*self.tau*self.model[0](x) # +torch.sum(x_ch1,dim=1,keepdim=True)/self.in_channels-self.out_channels*self.tau*(src-tgt)*T_x1
            out1 = out[:, :self.out_channels//2, :, :]       
            out2 = out[:, self.out_channels//2:, :, :]                                           # +torch.sum(x_ch2,dim=1,keepdim=True)/self.in_channels-self.out_channels*self.tau*(src-tgt)*T_x2
            out1=out1+torch.sum(x_ch1,dim=1,keepdim=True)/self.in_channels-(self.out_channels*self.tau*(src-tgt)*T_x1)/self.in_channels
            out2=out2+torch.sum(x_ch2,dim=1,keepdim=True)/self.in_channels-(self.out_channels*self.tau*(src-tgt)*T_x2)/self.in_channels
            out=torch.cat([out1,out2],dim=1)
            # 在第l=1个顺序分割层中求解方程（4.25）
        else:
            out=self.model[0](x)
        out=self.model[1](out)
        out=self.model[2](out)

        for i in np.arange(self.times-1): # times = 3，这里减1是因为第一层已经通过了，所以只需要再通过2层, i = 0,1
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
                
            # 在第l=2, 3个顺序分割层中求解方程（4.25）
            out=self.model[i*3+4](out)

        
        return out 
    

class MGPCConv_down(POTTS):
# the rest of the grid levels of the left branch
    def __init__(self,args=None, times=2, in_channels=32,out_channels=32,output_level=1): # in_channels和out_channels指的是与u1或u2做卷积的单个卷积核的输入和输出通道数
        super().__init__(args=args)

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.output_level=output_level
        self.times=times
        self.kernel_size=np.minimum(self.kernel_size_max//(3**output_level),self.kernel_size_bound)
        # self.pooling=nn.MaxPool2d(kernel_size=2,stride=2)

        self.model=nn.ModuleList()
        self.model.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,
                      stride=1,padding=1,bias=False)) # 用于接收u1,u2
        self.model.append(nn.BatchNorm2d(out_channels,affine=self.BNLearn)) # out_channels = 32，u1,u2的归一化层，至此，l=1的顺序分割层搭建完成
        self.model.append(nn.LeakyReLU(0.2))

        for i in np.arange(self.times-1): # times = 3，这里减1是因为第一层已经添加了，所以只需要再添加2层, i = 0,1, 对应l=2,3两个顺序分割层
            self.model.append(nn.Conv2d(in_channels=self.out_channels,out_channels=self.out_channels, 
                      kernel_size=self.kernel_size,
                      stride=1,padding=1,bias=False)) # 用于接收处理过后的u1,u2的out_channels个分量
            self.model.append(nn.BatchNorm2d(self.out_channels,affine=self.BNLearn))
            self.model.append(nn.LeakyReLU(0.2))
            '''
            in_channels = out_channels = 32,这里的in_channels数量等于l-1级顺序分割网络的输出通道数, 相当于c_{j,l}, 
            out_channels = 32, 也等于l+1级顺序分割网络的in_channels数量, kernel_size = 5
            至此，l=2,3的顺序分割层搭建完成
            '''
            
    def forward(self,x,src,tgt): # 在配准任务中不需要flist
        T_x1, T_x2 = torch.gradient(src, dim=(2, 3)) # 计算src的梯度，T_x1是x1方向的梯度，T_x2是x2方向的梯度
        x=x.float()
        C = x.shape[1]  # 通道数
        x_ch1 = x[:, :C//2, :, :]  # 前一半通道
        x_ch2 = x[:, C//2:, :, :]  # 后一半通道
        if self.tau_explicit:
            out=self.out_channels*self.tau*self.model[0](x)
            out1 = out[:, :self.out_channels//2, :, :]
            out2 = out[:, self.out_channels//2:, :, :]
            out1=out1+torch.sum(x_ch1,dim=1,keepdim=True)/self.in_channels-(self.out_channels*self.tau*(src-tgt)*T_x1)/self.in_channels
            out2=out2+torch.sum(x_ch2,dim=1,keepdim=True)/self.in_channels-(self.out_channels*self.tau*(src-tgt)*T_x2)/self.in_channels
            out=torch.cat([out1,out2],dim=1)
            
            # 在第l=1个顺序分割层中求解方程（4.25）
        else:
            out=self.model[0](x)
        out=self.model[1](out)
        out=self.model[2](out)

        for i in np.arange(self.times-1): # times = 3，这里减1是因为第一层已经通过了，所以只需要再通过2层, i = 0,1
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
            # 在第l=2, 3个顺序分割层中求解方程（4.25）
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
                      stride=1,padding=1,bias=False)) # 用于接收u1,u2
        self.model.append(nn.BatchNorm2d(self.out_channels,affine=self.BNLearn)) # out_channels = 32，u1,u2的归一化层，至此，l=1的顺序分割层搭建完成
        self.model.append(nn.LeakyReLU(0.2))

        for i in np.arange(self.times-1): # times = 3，这里减1是因为第一层已经添加了，所以只需要再添加2层, i = 0,1, 对应l=2,3两个顺序分割层
            self.model.append(nn.Conv2d(in_channels=self.out_channels,out_channels=self.out_channels, 
                      kernel_size=self.kernel_size,
                      stride=1,padding=1,bias=False)) # 用于接收处理过后的u1的out_channels个分量
            self.model.append(nn.BatchNorm2d(self.out_channels,affine=self.BNLearn))
            self.model.append(nn.LeakyReLU(0.2))
            '''
            in_channels = out_channels = 32,这里的in_channels数量等于l-1级顺序分割网络的输出通道数, 相当于c_{j,l}, 
            out_channels = 32, 也等于l+1级顺序分割网络的in_channels数量, kernel_size = 5
            至此，l=2,3的顺序分割层搭建完成
            '''
            
    def forward(self,x,src,tgt): # 在配准任务中不需要flist
        T_x1, T_x2 = torch.gradient(src, dim=(2, 3)) # 计算src的梯度，T_x1是x1方向的梯度，T_x2是x2方向的梯度
        C = x.shape[1]  # 通道数
        x_ch1_1 = x[:, :C//4, :, :]  # 前一半通道
        x_ch2_1 = x[:, C//4:C//2, :, :]  # 后一半通道
        x_ch1_2 = x[:, C//2:3*C//4, :, :]  # 前一半通道
        x_ch2_2 = x[:, 3*C//4:, :, :]  # 后一半通道
        x_ch1=torch.cat([x_ch1_1,x_ch1_2],dim=1) # (B,out_channels,H,W), u1部分
        x_ch2=torch.cat([x_ch2_1,x_ch2_2],dim=1) # (B,out_channels,H,W), u2部分
        if self.tau_explicit:
            out=self.out_channels*self.tau*self.model[0](x)
            out1 = out[:, :self.out_channels//2, :, :]
            out2 = out[:, self.out_channels//2:, :, :]
            out1=out1+torch.sum(x_ch1,dim=1,keepdim=True)/self.in_channels-(self.out_channels*self.tau*(src-tgt)*T_x1)/self.in_channels
            out2=out2+torch.sum(x_ch2,dim=1,keepdim=True)/self.in_channels-(self.out_channels*self.tau*(src-tgt)*T_x2)/self.in_channels
            out=torch.cat([out1,out2],dim=1)

            # 在第l=1个顺序分割层中求解方程（4.25）
        else:
            out=self.model[0](x)
        out=self.model[1](out)
        out=self.model[2](out)

        for i in np.arange(self.times-1): # times = 3，这里减1是因为第一层已经通过了，所以只需要再通过2层, i = 0,1
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
            # 在第l=2, 3个顺序分割层中求解方程（4.25）
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
        self.pooling=nn.MaxPool2d(kernel_size=2,stride=2) # 下采样
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') # 上采样 
        # out_channels = mid_channels[0] = 32, times = times_list[0]=3, kernel_size = kernel_size_max=243
        # self.times_list = {L_1,L_2,L_3,L_4,L_5} = [3,3,3,5,5], self.mid_channels = {c_1,c_2,c_3,c_4,c_5} = [32,32,64,128,256]
        for i in np.arange(self.level_max-1): # level_max = 5, i = 0,1,2,3, 减1是因为第一个网格级别已经添加了
            self.downs.append(MGPCConv_down(args=args, times=self.times_list[i+1], in_channels=self.mid_channels[i],
                                             out_channels=self.mid_channels[i+1],output_level=i+1,))
            # 后一个网格级别的in_channels等于前一个网格级别的out_channels
        
        self.ups=nn.ModuleList()    
        # self.combineWeight=nn.Parameter(torch.randn(self.level_max-1,2), requires_grad=True)
        for i in np.flip(np.arange(self.level_max-1)): # level_max = 5, i = 3,2,1,0，减1是因为第一个网格级别已经添加了
            self.ups.append(MGPCConv_up(args=args, times=self.times_list[i], in_channels=self.mid_channels[i] + self.mid_channels[i+1],
                                             out_channels=self.mid_channels[i],output_level=i,))
            
        self.final=nn.Conv2d(in_channels=self.mid_channels[0]+2,out_channels=2,kernel_size=3, stride=1, padding=1,
                      bias=True) # 用于接收u1
        
        

    def forward(self,x,src, tgt): # 在配准任务中不需要flist
        out=x
        x=x.float()
        # x_enc = [x] # 将x添加到x_enc列表中
        connect=[]
        for i in np.arange(self.level_max): # level_max = 5, i = 0,1,2,3,4

            out=self.downs[i](out,src,tgt) # out是左分支的第i个网格级别j的输出
            if i<self.level_max-1: # level_max-1 = 4, i = 0,1,2,3
                # x_enc.append(out) # 将左分支的每一个网格级别j的输出out添加到x_enc列表中
                connect.append(out) # 将左分支的每一个网格级别j的输出out添加到connect列表中
                out=self.pooling(out) # 下采样
                src = self.pooling(src) # 下采样
                tgt = self.pooling(tgt) # 下采样
            

        # out是左分支的最后一个网格级别j的输出
        connect=connect[::-1]

        for i in np.arange(self.level_max-1): # level_max-1 = 4, i = 0,1,2,3
            out=self.upsample(out) # 上采样
            src=self.upsample(src) # 上采样
            tgt=self.upsample(tgt) # 上采样
            out = torch.cat([out, connect[i]], dim=1)
            out=self.ups[i](out,src,tgt) # out是右分支的第i个网格级别j的输出

        if self.tau_explicit:
            out = torch.cat([out,x], dim=1) # out是右分支的第i个网格级别j的输出
            C = out.shape[1]  # 通道数
            # C2 = x.shape[1]  # 通道数
            
            out1_temp = out[:, :C//2, :, :]
            out2_temp = out[:, C//2:, :, :]
            # out=torch.cat([out,x],dim=1) # out是右分支的第i个网格级别j的输出
            out=self.final(out) # out是形变场的输出
            out1 = out[:, 0:1, :, :]
            out2 = out[:, 1:2, :, :]
            out1=out1+torch.sum(out1_temp,dim=1,keepdim=True)/self.mid_channels[0]
            out2=out2+torch.sum(out2_temp,dim=1,keepdim=True)/self.mid_channels[0]
            # 在第l=1个顺序分割层中求解方程（4.25）
        else:
            out=self.final(out)
        
        # out1=self.LaplaceAct(out1)
        # out2=self.LaplaceAct(out2)

        out=torch.cat([out1,out2],dim=1)


        return out
    
class layer1(POTTS):
# layer1是对输入的图像对进行预处理，从而初始化形变场
    def __init__(self, args=None, in_channels=1):
        super(layer1,self).__init__(args=args)
        self.dim=args.dim
        self.bn=args.bn
        
        
        self.layer1=nn.ModuleList()
        self.layer1.append(self.conv_block(dim=self.dim, in_channels=2, out_channels=2, batchnorm=self.bn))
        # layer1负责对输入的图像对进行预处理，从而初始化形变场
    
    def forward(self,x):
        x = x.float()
        out=self.layer1[0](x)
        return out # out是形变场的初始值


class POTTSNET(POTTS):
# assemble blocks
    def __init__(self, args=None, in_channels=1):
        super(POTTSNET,self).__init__(args=args)
        self.dim=args.dim
        self.bn=args.bn
        
        self.layer1=nn.ModuleList()
        self.layer1=layer1(args=args) # layer1是对输入的图像对进行预处理，从而初始化形变场
        

        self.blocks = nn.ModuleList() # 用于存储编码器的每一层
        for i in np.arange(self.num_blocks): # num_blocks = 4, i = 0,1,2,3
            self.blocks.append(Block(args=args))

    def forward(self, src, tgt):

        x = torch.cat([src, tgt], dim=1) # 拼接浮动图像和目标图像, dim=1表示在通道维度上拼接
        
        out=self.layer1(x) # x是原始图像对, layer1是对其进行预处理，out是形变场的初始值
        for idx in range(self.num_blocks):
            out=self.blocks[idx](out,src,tgt) # out是形变场的输出

        return out # out是形变场的最终值
    

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        src=src.float()
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
            new_locs = new_locs.float()
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
            new_locs = new_locs.float()

        return F.grid_sample(src, new_locs, mode=self.mode)




