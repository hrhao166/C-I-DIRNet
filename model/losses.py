"""
*Preliminary* pytorch implementation.

Losses for VoxelMorph
"""

import math
import torch
import numpy as np
from model.config import args
import torch.nn.functional as F


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
    计算张量 s 的二阶差分损失，输出为一个标量。
    假设 s 的维度为 (batch, channels, height, width)。
    """
    # 沿着 height 方向的二阶差分: s[i+1] - 2*s[i] + s[i-1]
    dyy = s[:, :, 2:, :] - 2.0 * s[:, :, 1:-1, :] + s[:, :, :-2, :]

    # 沿着 width 方向的二阶差分: s[j+1] - 2*s[j] + s[j-1]
    dxx = s[:, :, :, 2:] - 2.0 * s[:, :, :, 1:-1] + s[:, :, :, :-2]

    if penalty == 'l2':
        # 如果需要 L2 范数，就对差分结果进行平方
        dyy = dyy ** 2
        dxx = dxx ** 2
    else:
        # 否则可以当成 L1 范数，对差分结果取绝对值
        dyy = torch.abs(dyy)
        dxx = torch.abs(dxx)

    # 将两个方向的损失分别求平均后相加，并做一个简单的归一化(除以2)
    d = torch.mean(dyy) + torch.mean(dxx)
    return d / 2.0

def curl_regularizer(s):
    """
    计算 R1(u) = ∫ (∂₁u₁ - ∂₂u₂)² + (∂₂u₁ + ∂₁u₂)² dx
    输入 s: (B, 2, H, W), 返回标量。
    """
    # 拆通道
    u1 = s[:, 0, :, :]  # (B, H, W)
    u2 = s[:, 1, :, :]  # (B, H, W)

    # 一阶差分
    du1_dx1 = u1[:, :, 1:] - u1[:, :, :-1]   # ∂u1/∂x1, shape (B, H, W-1)
    du1_dx2 = u1[:, 1:, :] - u1[:, :-1, :]   # ∂u1/∂x2, shape (B, H-1, W)
    du2_dx1 = u2[:, :, 1:] - u2[:, :, :-1]   # ∂u2/∂x1
    du2_dx2 = u2[:, 1:, :] - u2[:, :-1, :]   # ∂u2/∂x2

    # 为了让两项同处于内部区域，裁剪到 (H-1, W-1)
    # 对于 dx1 方向要去掉最后一行：du*_dx1[:, :-1, :]
    # 对于 dx2 方向要去掉最后一列：du*_dx2[:, :, :-1]
    a = du1_dx1[:, :-1, :] - du2_dx2[:, :, :-1]  # (∂₁u₁ - ∂₂u₂)
    b = du1_dx2[:, :, :-1] + du2_dx1[:, :-1, :]  # (∂₂u₁ + ∂₁u₂)

    # 平均并返回
    loss = (a**2 + b**2).mean()
    return loss

def count_grid_folds(s):
    """
    计算批量位移场 s=(B,2,H,W) 的网格重叠数（det J_phi <= 0 的像素数）。
    返回形状 (B,) 的整型张量。
    """
    # 拆通道
    u1 = s[:, 0]  # (B,H,W)
    u2 = s[:, 1]  # (B,H,W)

    # 前向差分
    du1_dx1 = u1[:, :, 1:] - u1[:, :, :-1]   # shape (B,H,W-1)
    du1_dx2 = u1[:, 1:, :] - u1[:, :-1, :]   # shape (B,H-1,W)
    du2_dx1 = u2[:, :, 1:] - u2[:, :, :-1]
    du2_dx2 = u2[:, 1:, :] - u2[:, :-1, :]

    # 为了对齐到同一个 (H-1) x (W-1) 区域：
    # - du*_dx1 需要去掉最后一行  ⇒ [:, :-1, :]
    # - du*_dx2 需要去掉最后一列  ⇒ [:, :, :-1]
    J11 = 1.0 + du1_dx1[:, :-1, :]       # 1 + ∂u1/∂x1
    J12 =       du1_dx2[:, :, :-1]      #     ∂u1/∂x2
    J21 =       du2_dx1[:, :-1, :]      #     ∂u2/∂x1
    J22 = 1.0 + du2_dx2[:, :, :-1]      # 1 + ∂u2/∂x2

    # 计算行列式
    detJ = J11 * J22 - J12 * J21         # shape (B, H-1, W-1)

    # 统计每个样本中 detJ <= 0 的数量
    folds_per_batch = torch.sum(detJ <= 0.0, dim=[1,2])  # shape (B,)

    return folds_per_batch


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
    输入大小是[B,C,D,W,H]格式的，在计算ncc时用卷积来实现指定窗口内求和
    '''
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    # sum_filt = torch.ones([1, 1, *win]).to("cuda:{}".format(args.gpu))
    sum_filt = torch.ones([1, 1, *win]).to("cpu")
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


def cc_loss(x, y):
    # 根据互相关公式进行计算
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
