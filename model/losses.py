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

