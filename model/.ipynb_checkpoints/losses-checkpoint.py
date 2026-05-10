"""
*Preliminary* pytorch implementation.

Losses for VoxelMorph
"""

import math
import torch
import numpy as np
from model.config import args
import torch.nn.functional as F
import scipy.io as sio
import os
import random


def gradient_loss(s, penalty='l2'):
    
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
    

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)   # + torch.mean(dz)
    return d / 2.0


def second_order_loss(s, penalty='l2'):
    """
    Compute the second-order difference loss of tensor s and output a scalar.
    Assume the dimensions of s are (batch, channels, height, width).
    """
    # Second-order difference along the height direction: s[i+1] - 2*s[i] + s[i-1]
    dyy = s[:, :, 2:, :] - 2.0 * s[:, :, 1:-1, :] + s[:, :, :-2, :]

    # Second-order difference along the width direction: s[j+1] - 2*s[j] + s[j-1]
    dxx = s[:, :, :, 2:] - 2.0 * s[:, :, :, 1:-1] + s[:, :, :, :-2]

    if penalty == 'l2':
        # If the L2 norm is needed, square the difference result
        dyy = dyy ** 2
        dxx = dxx ** 2
    else:
        # Otherwise, treat it as the L1 norm and take the absolute value of the difference result
        dyy = torch.abs(dyy)
        dxx = torch.abs(dxx)

    # Average the losses in the two directions separately, add them, and apply a simple normalization (divide by 2)
    d = torch.mean(dyy) + torch.mean(dxx)
    return d / 2.0

def curl_regularizer(s):
    """
    Compute R1(u) = integral (d1u1 - d2u2)^2 + (d2u1 + d1u2)^2 dx
    Input s: (B, 2, H, W), return a scalar.
    """
    # Split channels
    u1 = s[:, 0, :, :]  # (B, H, W)
    u2 = s[:, 1, :, :]  # (B, H, W)

    # First-order difference
    du1_dx1 = u1[:, :, 1:] - u1[:, :, :-1]   # ∂u1/∂x1, shape (B, H, W-1)
    du1_dx2 = u1[:, 1:, :] - u1[:, :-1, :]   # ∂u1/∂x2, shape (B, H-1, W)
    du2_dx1 = u2[:, :, 1:] - u2[:, :, :-1]   # ∂u2/∂x1
    du2_dx2 = u2[:, 1:, :] - u2[:, :-1, :]   # ∂u2/∂x2

    # To keep both terms in the same interior region, crop to (H-1, W-1)
    # For the dx1 direction, remove the last row: du*_dx1[:, :-1, :]
    # For the dx2 direction, remove the last column: du*_dx2[:, :, :-1]
    a = du1_dx1[:, :-1, :] - du2_dx2[:, :, :-1]  # (∂₁u₁ - ∂₂u₂)
    b = du1_dx2[:, :, :-1] + du2_dx1[:, :-1, :]  # (∂₂u₁ + ∂₁u₂)

    # Average and return
    loss = (a**2 + b**2).mean()
    return loss

def count_grid_folds(s):
    """
    Compute the number of grid overlaps for batch displacement field s=(B,2,H,W) (number of pixels where det J_phi <= 0).
    Return an integer tensor with shape (B,).
    """
    # Split channels
    u1 = s[:, 0]  # (B,H,W), previously 0
    u2 = s[:, 1]  # (B,H,W), previously 1

    # Forward difference
    du1_dx1 = u1[:, :, 1:] - u1[:, :, :-1]   # shape (B,H,W-1)
    du1_dx2 = u1[:, 1:, :] - u1[:, :-1, :]   # shape (B,H-1,W)
    du2_dx1 = u2[:, :, 1:] - u2[:, :, :-1]
    du2_dx2 = u2[:, 1:, :] - u2[:, :-1, :]
    

    # To align to the same (H-1) x (W-1) region:
    # - du*_dx1 needs to remove the last row => [:, :-1, :]
    # - du*_dx2 needs to remove the last column => [:, :, :-1]
    J11 = 1.0 + du1_dx1[:, :-1, :]       # 1 + ∂u1/∂x1
    J12 =       du1_dx2[:, :, :-1]      #     ∂u1/∂x2
    J21 =       du2_dx1[:, :-1, :]      #     ∂u2/∂x1
    J22 = 1.0 + du2_dx2[:, :, :-1]      # 1 + ∂u2/∂x2

    # Compute the determinant
    detJ = J11 * J22 - J12 * J21         # shape (B, H-1, W-1)

    # Count the number of detJ <= 0 values in each sample
    folds_per_batch = torch.sum(detJ <= 0.0, dim=[1,2])  # shape (B,)

    return folds_per_batch


def mse_diff(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean squared error (MSE) between two tensors with shape (B, 2, W, H).
    
    Parameters:
        tensor1 (torch.Tensor): The first tensor, shape = (B, 2, W, H)
        tensor2 (torch.Tensor): The second tensor, shape = (B, 2, W, H)
    
    Returns:
        torch.Tensor: A scalar tensor representing the mean squared error over all elements.
    """
    # Check whether input dimensions are consistent
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"Input tensor shapes do not match: {tensor1.shape} vs {tensor2.shape}")
    
    # Compute the mean squared error and return a scalar
    return torch.mean((tensor1 - tensor2) ** 2)



def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def ncc_loss(I, J, win=None):
    '''
    The input size is in [B,C,D,W,H] format; when computing ncc, convolution is used to sum within the specified window
    '''
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).to("cuda:{}".format(args.gpu))
    # sum_filt = torch.ones([1, 1, *win]).to("cpu")
    pad_no = math.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    # print("I: ", I) # torch.Size([1, 1, 128, 128])
    # print("J: ", J) # torch.Size([1, 1, 128, 128])
    # print("I_var: ", I_var) # torch.Size([1, 1, 128, 128])
    # print("J_var: ", J_var) # torch.Size([1, 1, 128, 128])
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    # I=I.float()
    # J=J.float()
    filt = filt.float()
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross


def cc_loss(x, y):
    # Compute according to the cross-correlation formula
    dim = [2, 3, 4]
    mean_x = torch.mean(x, dim, keepdim=True)
    mean_y = torch.mean(y, dim, keepdim=True)
    mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
    mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
    stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
    stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
    return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


def Get_Ja(flow):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    return D1 - D2 + D3


def NJ_loss(ypred):
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5 * (torch.abs(Get_Ja(ypred)) - Get_Ja(ypred))
    return torch.sum(Neg_Jac)




def compute_local_stats(I, J, filt, stride, padding, win):
    """
    Compute statistics in the local window: mean, variance, covariance
    Args:
        I, J: Input tensors, shape (1, 2, W, H)
        filt: Convolution kernel, shape (1, in_channels, *win)
        stride: Convolution stride
        padding: Padding size
        win: Window size (spatial dimensions)
    """
    # Compute I^2, J^2, and I*J
    I2 = I.pow(2)
    J2 = J.pow(2)
    IJ = I * J

    # Local sum computation
    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

    # Compute the number of elements in the window (channels x product of spatial dimensions)
    in_channels = I.size(1)
    win_size = in_channels
    for w in win:
        win_size *= w

    # Compute means
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    # Compute covariance and variance
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I.pow(2) * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J.pow(2) * win_size

    return I_var, J_var, cross


def ncc_similarity(I, J, win=None):
    """
    Compute the normalized cross-correlation (NCC) similarity of two tensors
    Args:
        I, J: Input tensors, shape (1, 2, W, H)
        win: Window size (default [9,9])
    Returns:
        NCC similarity coefficient (scalar)
    """
    # Determine the spatial dimensions and set the default window
    ndims = len(I.shape) - 2  # Input shape is (1,2,W,H), subtracting the B and C dimensions
    assert ndims in [1, 2, 3], "Only 1D/2D/3D tensors are supported"
    if win is None:
        win = [9] * ndims

    # Create the convolution kernel (the number of input channels must match the tensor)
    in_channels = I.size(1)
    sum_filt = torch.ones([1, in_channels, *win], dtype=I.dtype, device=I.device)

    # Compute padding and stride
    pad_no = [math.floor(w/2) for w in win]
    stride = [1] * ndims
    padding = pad_no

    # Get local statistics
    I_var, J_var, cross = compute_local_stats(I, J, sum_filt, stride, padding, win)

    # Numerical-stability handling
    eps = 1e-5
    I_var = torch.clamp(I_var, min=0)  # Variance is non-negative
    J_var = torch.clamp(J_var, min=0)

    # Compute the normalized cross-correlation coefficient
    denominator = torch.sqrt(I_var * J_var) + eps
    cc = cross / denominator

    # Return the global average similarity
    return torch.mean(cc)

def field_ncc_loss(I, J, win=None):
    """
    Normalized cross-correlation loss function based on a local window (a smaller loss means higher similarity)
    Args:
        I, J: Input tensors, shape (1, 2, W, H)
        win: Window size (default [9,9])
    Returns:
        NCC loss value (scalar), roughly in the range [-1, 1] (actually mapped by the negative sign from [-1, 1] to [-1, 1])
    """
    # Compute the NCC similarity coefficient (range [-1, 1])
    similarity = ncc_similarity(I, J, win)
    
    # Convert similarity to loss (higher similarity -> lower loss)
    # Convert the similarity-maximization problem to a loss-minimization problem by negating it
    loss = -similarity
    
    # Optional: if the loss needs to be non-negative, add an offset (adjust according to actual needs)
    # loss = 1 - similarity  # At this point the loss range is [0, 2]
    
    return loss

# Keep the original compute_local_stats and ncc_similarity functions unchanged





    
