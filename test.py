# python imports
import os
import glob
import math
import h5py
import random
import torch.nn as nn
import warnings
import time
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List
# external imports
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from scipy.io import savemat
# import SimpleITK as sitk
from PIL import Image

import torch.utils.data as Data
# internal imports
from model import losses
from model.config import args
# from model.datagenerators import Dataset
from model.PottsMorph_model import POTTSNET, SpatialTransformer


# -------------sup------------------------------

# -----------OASIS----------------
mask_result_dir = r"./Result/sup/OASIS/mask"
field_result_dir = r"./Result/sup/OASIS/deformation_field"
img_pair_dir = r"./Result/sup/OASIS/image_pair"
# -----------OASIS----------------


# # -----------OASIS_64----------------
# mask_result_dir = r"./Result/sup/OASIS_64/mask"
# field_result_dir = r"./Result/sup/OASIS_64/deformation_field"
# img_pair_dir = r"./Result/sup/OASIS_64/image_pair"
# # -----------OASIS_64----------------

# # -----------OASIS_256----------------
# mask_result_dir = r"./Result/sup/OASIS_256/mask"
# field_result_dir = r"./Result/sup/OASIS_256/deformation_field"
# img_pair_dir = r"./Result/sup/OASIS_256/image_pair"
# # -----------OASIS_256----------------

# # -----------ACDC----------------
# mask_result_dir = r"./Result/sup/ACDC/mask"
# field_result_dir = r"./Result/sup/ACDC/deformation_field"
# img_pair_dir = r"./Result/sup/ACDC/image_pair"
# # -----------ACDC----------------

# # -----------CAMUS----------------
# mask_result_dir = r"./Result/sup/CAMUS/mask"
# field_result_dir = r"./Result/sup/CAMUS/deformation_field"
# img_pair_dir = r"./Result/sup/CAMUS/image_pair"
# # -----------CAMUS----------------

# # -----------addition----------------
# mask_result_dir = r"./Result/addition"
# field_result_dir = r"./Result/addition"
# img_pair_dir = r"./Result/addition"
# # -----------addition----------------




# -------------sup------------------------------


# -------------unsup------------------------------

# # -----------OASIS----------------
# mask_result_dir = r"./Result/unsup/OASIS/mask"
# field_result_dir = r"./Result/unsup/OASIS/deformation_field"
# img_pair_dir = r"./Result/unsup/OASIS/image_pair"
# # -----------OASIS----------------

# # -----------ACDC----------------
# mask_result_dir = r"./Result/unsup/ACDC/mask"
# field_result_dir = r"./Result/unsup/ACDC/deformation_field"
# img_pair_dir = r"./Result/unsup/ACDC/image_pair"
# # -----------ACDC----------------

# # -----------CAMUS----------------
# mask_result_dir = r"./Result/unsup/CAMUS/mask"
# field_result_dir = r"./Result/unsup/CAMUS/deformation_field"
# img_pair_dir = r"./Result/unsup/CAMUS/image_pair"
# # -----------CAMUS----------------


# -------------unsup------------------------------

# translate all chinese comments to english

def save_image(img, name, slice_idx=None):
    # ensure the save directory exists
    os.makedirs(img_pair_dir, exist_ok=True)

    # convert to NumPy format
    img = img[0, 0, ...].cpu().detach().numpy()

    # if it's 3D data, select a slice
    if img.ndim == 3:  # (D, H, W)
        if slice_idx is None:
            slice_idx = img.shape[0] // 2  # select the middle slice
        img = img[slice_idx]  # extract the (H, W) slice

    # Normalize to 0-255
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    img = img.astype(np.uint8)

    # Save PNG
    img_pil = Image.fromarray(img).convert("L")
    img_pil.save(os.path.join(img_pair_dir, f"{name}.jpg"))


def read_one_image(folder_path):
    """
    Read a PNG image from the specified folder randomly and return a normalized tensor.

    Parameters:
        folder_path (str): The path to the folder containing PNG images.

    Returns:
        torch.Tensor: The normalized image tensor with shape (1, 1, H, W), values in [0,1].

    Raises:
        ValueError: If no PNG images are found in the folder.
    """
    # Get a list of all PNG files in the folder
    image_files = [os.path.join(folder_path, f) 
                   for f in os.listdir(folder_path) 
                   if f.lower().endswith('.png')]
    
    # Check if at least one PNG image is found
    if not image_files:
        raise ValueError("No PNG images found in the folder!")
    
    # Randomly select an image file
    selected_file = random.choice(image_files)
    
    # Read the image and convert to grayscale
    image = Image.open(selected_file).convert("L")
    
    # Define the transformation pipeline: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])
    
    normalized_image = transform(image)  # Shape: (1, H, W), values in [0,1]
    normalized_image = normalized_image.unsqueeze(0)  # Shape becomes (1, 1, H, W)

    return normalized_image

# -----------------------Read test data with masks------------------------------------------------------

def load_mat_pairs(folder_path: str):
    """
    Read all .mat files under folder_path and return four lists:
      - T_list, D_list:      List[torch.FloatTensor], each shape (1,1,128,128)
      - T_mask_list, D_mask_list: List[torch.BoolTensor], each shape (1,1,128,128)
    The elements at the same index correspond to the T, D, T_mask, D_mask in the same mat file.
    """
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),  # HxW -> 1xHxW, float in [0,1]
    ])
    
    T_list = []
    D_list = []
    T_mask_list = []
    D_mask_list = []
    
    # Traverse all .mat files (sorted by filename)
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith('.mat'):
            continue
        mat_path = os.path.join(folder_path, fname)
        
        data = sio.loadmat(mat_path)

        T_np      = data['T']
        D_np      = data['D']
        T_mask_np = data['T_mask']
        D_mask_np = data['D_mask']
        
        T_tensor = transform(T_np).unsqueeze(0)
        D_tensor = transform(D_np).unsqueeze(0)
        
        T_mask_tensor = torch.from_numpy((T_mask_np.astype(bool))).unsqueeze(0).unsqueeze(0)
        D_mask_tensor = torch.from_numpy((D_mask_np.astype(bool))).unsqueeze(0).unsqueeze(0)
        
        # Append to lists
        T_list.append(T_tensor)
        D_list.append(D_tensor)
        T_mask_list.append(T_mask_tensor)
        D_mask_list.append(D_mask_tensor)
    
    return T_list, D_list, T_mask_list, D_mask_list


# translate all chinese comments to english

def load_h5_4d(path_h5):
    """
    Read data from an HDF5 file and return four lists of tensors.

    Args:
        path_h5 (str): HDF5 file path

    Returns:
        T_list       (List[torch.FloatTensor]): diastolic image
        D_list       (List[torch.FloatTensor]): systolic image
        T_mask_list  (List[torch.BoolTensor]) : label_d mask
        D_mask_list  (List[torch.BoolTensor]) : label_s mask
    """
    T_list = []
    D_list = []
    T_mask_list = []
    D_mask_list = []

    with h5py.File(path_h5, 'r') as f:
        # ds_T  = f['diastolic']   # (N, 1, 128, 128)
        # ds_D  = f['systolic']    # (N, 1, 128, 128)
        # ds_Tm = f['label_d']     # (N, 1, 128, 128)
        # ds_Dm = f['label_s']     # (N, 1, 128, 128)
        ds_T  = f['T']   # (N, 1, 128, 128)
        ds_D  = f['D']    # (N, 1, 128, 128)
        ds_Tm = f['T_mask']     # (N, 1, 128, 128)
        ds_Dm = f['D_mask']     # (N, 1, 128, 128)

        N = ds_T.shape[0]
        for i in range(N):
            # Image: float32 -> (1, 128, 128) -> (1, 1, 128, 128)
            t  = torch.from_numpy(ds_T[i].astype(np.float32)).unsqueeze(0)
            d  = torch.from_numpy(ds_D[i].astype(np.float32)).unsqueeze(0)

            # Mask: bool type -> (1, 1, 128, 128)
            tm = torch.from_numpy((ds_Tm[i] == 1).astype(bool)).unsqueeze(0).to(torch.bool)
            dm = torch.from_numpy((ds_Dm[i] == 1).astype(bool)).unsqueeze(0).to(torch.bool)

            T_list.append(t)
            D_list.append(d)
            T_mask_list.append(tm)
            D_mask_list.append(dm)

    return T_list, D_list, T_mask_list, D_mask_list


# -----------------------Read test data with masks------------------------------------------------------

# -----------------------Calculate Re-SSD------------------------------------------------------

def ssd(imgA: np.ndarray, imgB: np.ndarray) -> float:
    """
    Calculate SSD(A, B) = (1/2) * sum((A_ij - B_ij)^2).
    """
    A = imgA.astype(np.float64)
    B = imgB.astype(np.float64)
    diff = A - B
    return 0.5 * np.sum(diff**2)

def compute_re_ssd(f_img: np.ndarray, m_img: np.ndarray, m2f_img: np.ndarray) -> float:
    """
    Calculate Re-SSD(m, f) = SSD(m2f, f) / SSD(m, f).
    """
    denom = ssd(m_img, f_img)
    if denom == 0:
        # If the two images are identical or both constant images, the denominator might be 0
        return float('inf')
    
    numer = ssd(m2f_img, f_img)
    return numer / denom

# -----------------------Calculate Re-SSD------------------------------------------------------

# -----------------------Calculate Dice---------------------------------------------------------

def dice_score(pred: torch.Tensor,
               target: torch.Tensor,
               eps: float = 1e-6) -> torch.Tensor:
    """
    Calculate the Dice coefficient between two binary masks.

    Parameters:
        pred   (torch.Tensor): Predicted mask, shape (1,1,H,W), dtype=torch.bool or {0,1} float/int
        target (torch.Tensor): Ground truth mask, shape (1,1,H,W), dtype=torch.bool or {0,1} float/int
        eps     (float): Small constant to prevent division by zero

    Returns:
        torch.Tensor: Scalar, Dice coefficient in [0,1]
    """
    # Reshape inputs to 1D
    pred_flat   = pred.reshape(-1).float()
    target_flat = target.reshape(-1).float()

    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    # Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    dice = (2.0 * intersection + eps) / (union + eps)

    return dice

# -----------------------Calculate Dice---------------------------------------------------------



# translate all chinese comments to english

# -----------------------Compose deformation fields------------------------------------------------------

# translate all chinese to english

def compose_displacements(u1: torch.Tensor,
                          u2: torch.Tensor) -> torch.Tensor:
    """
    Compose two displacement fields u1, u2: first transform by u1, then by u2.
    Here we generate the base_grid[0,0,:,:] = row indices (0,1,2,...,H-1)
                     base_grid[0,1,:,:] = column indices (0,1,2,...,W-1)
    The final output is also a displacement field of shape [B,2,H,W].
    """
    B, C, H, W = u1.shape
    assert C == 2, "displacement fields must have 2 channels (y and x displacements)"

    # 1) Construct base_grid, shape = [B, 2, H, W]
    ys = torch.arange(0, H, device=u1.device, dtype=u1.dtype)
    xs = torch.arange(0, W, device=u1.device, dtype=u1.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    
    base_grid = torch.stack((grid_y, grid_x), dim=0)       # [2, H, W]
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)   # [B, 2, H, W]

    #      pts[:,0,:,:] = y + u1_y
    #      pts[:,1,:,:] = x + u1_x
    pts = base_grid + u1

    pts = pts.permute(0, 2, 3, 1)  # [B, H, W, 2],  now pts[...,0]=y, pts[...,1]=x

    x = pts[..., 1]
    y = pts[..., 0]
    x_norm = 2.0 * (x / (W - 1) - 0.5)
    y_norm = 2.0 * (y / (H - 1) - 0.5)
    grid_norm = torch.stack((x_norm, y_norm), dim=-1)  # [B, H, W, 2]

    
    u2_warped = F.grid_sample(
        u2,             # [B, 2, H, W]
        grid_norm,      # [B, H, W, 2]
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )  # => [B, 2, H, W]

    u_comp = u1 + u2_warped
    return u_comp





def compose_displacements_list(u_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Compose all displacement fields in a list sequentially.
    
    Parameters:
      u_list: List of tensors, each of shape [B,2,H,W], representing displacement fields to be composed in order.
    Returns:
      u_total: Tensor of shape [B,2,H,W],representing the total composed displacement field.
    """
    if not u_list:
        raise ValueError("u_list must contain at least one displacement field")
    
    u_total = u_list[0]
    for u in u_list[1:]:
        u_total = compose_displacements(u_total, u)
    
    return u_total



# -----------------------Compose deformation fields------------------------------------------------------


# -----------------------Plot deformation fields------------------------------------------------------

def plot_deformation_grid_batch(u_tot, stride=3, batch_idx=0, index=0):
    u_tot[:, :, 0, :] = 0
    # bottom row
    u_tot[:, :, -1, :] = 0
    # left column
    u_tot[:, :, :, 0] = 0
    # right column
    u_tot[:, :, :, -1] = 0

    
    if hasattr(u_tot, "cpu"):
        u_tot = u_tot.detach().cpu().numpy()

    B, C, H, W = u_tot.shape
    assert 0 <= batch_idx < B, f"batch_idx must be in [0, {B-1}]"
    assert C == 2, "channels must be 2 (x and y displacements)"

    flow = u_tot[batch_idx]               # (2, H, W)
    flow = np.transpose(flow, (1, 2, 0))  # (H, W, 2)

    x = np.linspace(1, W, W)      # 1, 2, ..., W
    y = np.linspace(1, H, H)      # 1, 2, ..., H
    x_grid, y_grid = np.meshgrid(x, y, indexing='xy')  # (H, W)

    x_disp = flow[..., 0]
    y_disp = flow[..., 1]

    phyx = x_grid + x_disp
    phyy = y_grid + y_disp

    dpi = 50
    plt.figure(figsize=(3*W / dpi, 3*H / dpi), dpi=dpi)

    for row in range(0, H, stride):
        plt.plot(phyx[row, :], phyy[row, :], 'b-', linewidth=3)

    for col in range(0, W, stride):
        plt.plot(phyx[:, col], phyy[:, col], 'b-', linewidth=3)

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')  
    ax.set_xlim(1, W)
    ax.set_ylim(H, 1)  

    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.xticks([]); plt.yticks([])
    plt.margins(0)
    ax.set_position([0, 0, 1, 1])

    out_path = os.path.join(field_result_dir, f"{index}_field.jpg")
    plt.savefig(out_path, format='jpg', bbox_inches='tight', pad_inches=0)
    plt.close()




# -----------------------Plot deformation fields------------------------------------------------------

# -----------------------Plot masks--------------------------------------------------------

# translate all chinese to english

def plot_masks(mask1: torch.Tensor,
               mask2: torch.Tensor,
               mask3: torch.Tensor,
               index: int,
               titles=None):
    """
    Save three (1,1,H,W) boolean masks as three EPS images respectively.

    Parameters:
        mask1, mask2, mask3 (torch.Tensor): dtype=torch.bool, shape=(1,1,H,W)
        i (int): Index for generating file names
        titles (list of str, optional): Titles for the three images, if None, no titles will be set
    """
    # Input validation
    for idx, m in enumerate((mask1, mask2, mask3), 1):
        if not (isinstance(m, torch.Tensor) and m.dtype == torch.bool):
            raise ValueError(f"mask{idx} must be a torch.bool type Tensor")
        if m.ndim != 4 or m.shape[0] != 1 or m.shape[1] != 1:
            raise ValueError(f"mask{idx} shape must be (1,1,H,W)")

    # Prepare titles
    if titles is not None:
        if len(titles) != 3:
            raise ValueError("titles length must be 3")

    # Remove batch and channel dimensions, convert to NumPy
    masks_np = [
        mask1.squeeze(0).squeeze(0).cpu().numpy(),
        mask2.squeeze(0).squeeze(0).cpu().numpy(),
        mask3.squeeze(0).squeeze(0).cpu().numpy(),
    ]

    # Prepare information for saving
    info = [
        ("m_mask",    masks_np[0], titles[0] if titles else None),
        ("m2f_mask",  masks_np[1], titles[1] if titles else None),
        ("f_mask",    masks_np[2], titles[2] if titles else None),
    ]

    # Ensure output directory exists
    os.makedirs(mask_result_dir, exist_ok=True)

    # Save each mask individually
    for name, mask_np, title in info:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        out_path = os.path.join(mask_result_dir, f"{name}_{index}.jpg")
        plt.savefig(out_path, format='jpg', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

# -----------------------Plot masks--------------------------------------------------------






def test():
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    
    h5_path = r".\data\test_data_with_label_64\test_pairs_half.h5"
    # "D:\PottsMorph_5\PottsMorph_5\data\test_data_with_label_64\test_pairs_half.h5"
    # "D:\PottsMorph_5\PottsMorph_5\data\test_data_with_label_256\test_pairs_double.h5"
    T_list, D_list, T_mask_list, D_mask_list = load_h5_4d(h5_path)
    # vol_size = T_list[0].shape[2:]
    vol_size = [128, 128]

    
    if args.train_mode:
        UNet_list = []
        for level in range(1, args.cascade_nums + 1):
            model = POTTSNET(args).to(device)
            checkpoint_path = getattr(args, f'supervised_checkpoint{level}_path')  
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            UNet_list.append(model)

        for model in UNet_list:
            model= model.float()  
            model.eval()
        
        UNet1, UNet2, UNet3, UNet4, UNet5, UNet6, UNet7, UNet8, UNet9, UNet10 = UNet_list
    else:
        UNet_list = []
        for level in range(1, args.cascade_nums + 1):
            model = POTTSNET(args).to(device)
            checkpoint_path = getattr(args, f'unsupervised_checkpoint{level}_path')  
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            UNet_list.append(model)

        for model in UNet_list:
            model= model.float()  
            model.eval()

        UNet1, UNet2, UNet3, UNet4, UNet5, UNet6, UNet7, UNet8, UNet9, UNet10 = UNet_list
        

    STN = SpatialTransformer(vol_size).to(device) 
    STN = STN.float()
    STN.eval()
    
    # translate all chinese to english
    
    # opt = Adam(UNet.parameters(), lr=args.lr)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn_1 = losses.second_order_loss
    grad_loss_fn_2 = losses.gradient_loss
    grid_folds = losses.count_grid_folds

   
    # Training loop.
    data_1 = [] 
    data_2 = [] 
    data_3 = [] 
    data_4 = [] 
    
    k = 0
    # translate all chinese to english
    for i in range(1,args.n_iter_in_test + 1): # 1,args.n_iter_in_test + 1
        # Generate the moving images and convert them to tensors.

        # ------------------------------------(MRI,128)----------------------------------------

        mat_dir = r"./data/test_data_with_label"
        T_list, D_list, T_mask_list, D_mask_list = load_mat_pairs(mat_dir)

        input_moving = T_list[i % len(T_list)].to(device).float()
        moving_label = T_mask_list[i % len(T_mask_list)].to(device)  

        input_fixed = D_list[i % len(D_list)].to(device).float()  
        fixed_label = D_mask_list[i % len(D_mask_list)].to(device)  

        moving_label_64 = moving_label.float() 

        input_moving_temp = input_moving.clone()

        # ------------------------------------(MRI,128)----------------------------------------

        # ------------------------------------(MRI,64,256)----------------------------------------

        # h5_path = r"D:\PottsMorph_5\PottsMorph_5\data\test_data_with_label_64\test_pairs_half.h5"
        # # "D:\PottsMorph_5\PottsMorph_5\data\test_data_with_label_256\test_pairs_double.h5"
        # # "D:\PottsMorph_5\PottsMorph_5\data\test_data_with_label_64\test_pairs_half.h5"
        # T_list, D_list, T_mask_list, D_mask_list = load_h5_4d(h5_path)

        # input_moving = T_list[i % len(T_list)].to(device).float()
        # moving_label = T_mask_list[i % len(T_mask_list)].to(device)  

        # input_fixed = D_list[i % len(D_list)].to(device).float()  
        # fixed_label = D_mask_list[i % len(D_mask_list)].to(device)  

        # moving_label_64 = moving_label.float()  

        # input_moving_temp = input_moving.clone()

        # ------------------------------------(MRI,64,256)----------------------------------------


        # ------------------------------------(ACDC, CAMUS)----------------------------------------

        # h5_path = r"./data/additional_data_with_mask/CAMUS/CAMUS2H_test_100.h5"
        # # r"./data/additional_data_with_mask/CAMUS/CAMUS2H_test_100.h5"
        # # r"./data/additional_data_with_mask/ACDC/acdc_test_54.h5"
        # T_list, D_list, T_mask_list, D_mask_list = load_h5_4d(h5_path)

        # input_moving = T_list[i % len(T_list)].to(device).float()
        # moving_label = T_mask_list[i % len(T_mask_list)].to(device)  

        # input_fixed = D_list[i % len(D_list)].to(device).float()  
        # fixed_label = D_mask_list[i % len(D_mask_list)].to(device)  

        # moving_label_64 = moving_label.float()  

        # input_moving_temp = input_moving.clone()

        # ------------------------------------(ACDC, CAMUS)----------------------------------------
       
        flow_list = []  
        MFN_list = []

        torch.cuda.synchronize()
        start=time.perf_counter()

        
        # translate all chinese to english
        for j in range(1, args.cascade_nums + 1): # 1, args.cascade_nums + 1
            UNet = eval(f'UNet{j}') # eval(f'UNet{j}')
            flow_m2f = UNet(input_moving, input_fixed)
            MFN = grid_folds(flow_m2f)  
            MFN = MFN.float()  
            MFN_list.append(MFN.item())  
            m2f = STN(input_moving, flow_m2f)
            input_moving = m2f  

            # -----------------------------
            
            m2f_label = STN(moving_label_64, flow_m2f)
            moving_label_64 = m2f_label  

            # -----------------------------

            flow_list.append(flow_m2f)
        
        
        torch.cuda.synchronize()
        end=time.perf_counter()

        flow_list.reverse()  
        flow_m2f = compose_displacements_list(flow_list) # compose_n_flows

        

        #---------------------------------------------------
        m2f_label = (m2f_label > 0.5) 
        #---------------------------------------------------


        # Calculate loss
        sim_loss = sim_loss_fn(m2f, input_fixed)
        primary_sim = sim_loss_fn(input_moving_temp, input_fixed)  
        
        grad_loss = grad_loss_fn_1(flow_m2f) + grad_loss_fn_2(flow_m2f)
        loss = sim_loss + args.alpha_1 * grad_loss
        MFN = grid_folds(flow_m2f)
        
        dicescore = dice_score(m2f_label, fixed_label)
        

        # Calculate Re-SSD
        re_ssd = compute_re_ssd(input_fixed[0, 0, ...].cpu().detach().numpy(),
                               input_moving_temp[0, 0, ...].cpu().detach().numpy(),
                               m2f[0, 0, ...].cpu().detach().numpy()) 
        
        MFN = sum(MFN_list)

        tt=end-start
        print("i: %d  loss: %f  sim: %f  primary_sim: %f  grad: %f  validation_time: %f  re_ssd: %f  MFN: %d  dice: %f"  % (i, loss.item(), sim_loss.item(),primary_sim.item(),grad_loss.item(), tt, re_ssd, MFN, dicescore), flush=True)

        
        
        if re_ssd != 'inf':
            data_1.append(re_ssd.item()) 
            data_2.append(dicescore.item())
            data_3.append(MFN)  
            if i >= 2:  
                data_4.append(tt)  
            
            # Save images
            m_name = str(i) + "_m"
            m2f_name = str(i) + "_m2f"
            f_name = str(i) + "_f"
            save_image(input_fixed, f_name)
            save_image(input_moving_temp, m_name)
            save_image(m2f, m2f_name)
            plot_deformation_grid_batch(flow_m2f, stride=3, batch_idx=0, index=i)  
            
            plot_masks(moving_label, m2f_label, fixed_label,index=i,
                    titles=["Moving Image Mask", "m2f Image Mask", "Fixed Image Mask"])
            
            print("warped images have saved.")
            k += 1

    print("total number of saved images: ", k)    
    

    average_re_ssd = np.mean(data_1)
    average_dice = np.mean(data_2)
    average_MFN = np.mean(data_3)
    average_tt = np.mean(data_4)

    std_re_ssd = np.std(data_1, ddof=1)
    std_dice = np.std(data_2, ddof=1)
    std_MFN = np.std(data_3, ddof=1)
    std_tt = np.std(data_4, ddof=1)

    print(f"\n average Re-SSD = {average_re_ssd:.6f}")
    print(f"\n average Dice = {average_dice:.6f}")
    print(f"\n average MFN = {average_MFN:.6f}")
    print(f"\n average time = {average_tt:.6f}")

    print(f"\n Re-SSD standard deviation = {std_re_ssd:.6f}")
    print(f"\n Dice standard deviation = {std_dice:.6f}")
    print(f"\n MFN standard deviation = {std_MFN:.6f}")
    print(f"\n time standard deviation = {std_tt:.6f}")


    
    

    

    
    
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    test()