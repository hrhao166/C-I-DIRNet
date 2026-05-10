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
from model.datagenerators import Dataset
from model.PottsMorph_model import POTTSNET, SpatialTransformer


# -------------sup------------------------------

# # -----------OASIS----------------
# mask_result_dir = r"./Result/sup/OASIS/mask"
# field_result_dir = r"./Result/sup/OASIS/deformation_field"
# img_pair_dir = r"./Result/sup/OASIS/image_pair"
# # -----------OASIS----------------

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

# -----------addition----------------
mask_result_dir = r"./Result/addition"
field_result_dir = r"./Result/addition"
img_pair_dir = r"./Result/addition"
# -----------addition----------------




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

def save_image(img, name, slice_idx=None):
    """
    Save medical images in PNG format. If it is a 3D image, save multiple PNG slices.
    
    :param img: Input image (Tensor), shape (B, C, D, H, W) or (B, C, H, W).
    :param name: Saved file name (without extension).
    :param slice_idx: If it is a 3D image, specify which slice to save (defaults to the middle slice).
    """
    # Ensure the save directory exists
    os.makedirs(img_pair_dir, exist_ok=True)

    # Convert to NumPy format
    img = img[0, 0, ...].cpu().detach().numpy()

    # If it is 3D data, select a slice
    if img.ndim == 3:  # (D, H, W)
        if slice_idx is None:
            slice_idx = img.shape[0] // 2  # Select the middle slice
        img = img[slice_idx]  # Extract the (H, W) slice

    # Normalize to 0-255
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    img = img.astype(np.uint8)

    # Save PNG
    img_pil = Image.fromarray(img).convert("L")
    img_pil.save(os.path.join(img_pair_dir, f"{name}.jpg"))


def read_one_image(folder_path):
    """
    Randomly read a PNG image from the specified folder and return an image tensor normalized to the [0,1] range.

    Parameters:
        folder_path (str): Folder path for storing PNG images

    Returns:
        torch.Tensor: Normalized image tensor, shape (1, 1, H, W), value range [0,1]

    Exceptions:
        ValueError: Raise an exception when there are no PNG images in the folder.
    """
    # Get the full path list of all PNG files in the folder
    image_files = [os.path.join(folder_path, f) 
                   for f in os.listdir(folder_path) 
                   if f.lower().endswith('.png')]
    
    # Check whether there is at least one PNG image
    if not image_files:
        raise ValueError("There are no PNG images in the folder!")
    
    # Randomly select an image file
    selected_file = random.choice(image_files)
    
    # Read the image and convert it to grayscale
    image = Image.open(selected_file).convert("L")
    
    # Define the transform pipeline: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Automatically convert to the [0,1] range, shape (1, H, W)
    ])
    
    # Apply transform
    normalized_image = transform(image)
    # Add the batch dimension so the shape becomes (1, 1, H, W)
    normalized_image = normalized_image.unsqueeze(0)  # Shape becomes (1, 1, H, W)

    return normalized_image

# -----------------------Read test data with masks------------------------------------------------------

def load_mat_pairs(folder_path: str):
    """
    Read all .mat files under folder_path and return four lists:
      - T_list, D_list:      List[torch.FloatTensor], each shape (1,1,128,128)
      - T_mask_list, D_mask_list: List[torch.BoolTensor], each shape (1,1,128,128)
    The four elements under the same index correspond to T, D, T_mask, and D_mask in the same mat file.
    """
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),  # HxW -> 1xHxW, float in [0,1]
    ])
    
    T_list = []
    D_list = []
    T_mask_list = []
    D_mask_list = []
    
    # Iterate over all .mat files (sorted by file name)
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith('.mat'):
            continue
        mat_path = os.path.join(folder_path, fname)
        
        data = sio.loadmat(mat_path)
        # Read the original array (128,128)
        T_np      = data['T']
        D_np      = data['D']
        T_mask_np = data['T_mask']
        D_mask_np = data['D_mask']
        
        # Convert to Tensor
        # T, D: ToTensor -> (1,H,W), then unsqueeze -> (1,1,H,W)
        T_tensor = transform(T_np).unsqueeze(0)
        D_tensor = transform(D_np).unsqueeze(0)
        
        # masks: first convert to bool numpy, then to BoolTensor, unsqueeze -> (1,1,H,W)
        T_mask_tensor = torch.from_numpy((T_mask_np.astype(bool))).unsqueeze(0).unsqueeze(0)
        D_mask_tensor = torch.from_numpy((D_mask_np.astype(bool))).unsqueeze(0).unsqueeze(0)
        
        # Add to the list
        T_list.append(T_tensor)
        D_list.append(D_tensor)
        T_mask_list.append(T_mask_tensor)
        D_mask_list.append(D_mask_tensor)
    
    return T_list, D_list, T_mask_list, D_mask_list


def load_h5_4d(path_h5):
    """
    Read the HDF5 file and take diastolic, systolic, label_d, and label_s with the same index
    Store them separately in four lists and return them.
    - Images are FloatTensor, shape=(1,1,128,128)
    - Masks are BoolTensor, shape=(1,1,128,128)

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
        ds_T  = f['diastolic']   # (N, 1, 128, 128)
        ds_D  = f['systolic']    # (N, 1, 128, 128)
        ds_Tm = f['label_d']     # (N, 1, 128, 128)
        ds_Dm = f['label_s']     # (N, 1, 128, 128)

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

# -----------------------Compute Re-SSD------------------------------------------------------

def ssd(imgA: np.ndarray, imgB: np.ndarray) -> float:
    """
    Compute SSD(A, B) = (1/2) * sum((A_ij - B_ij)^2).
    """
    A = imgA.astype(np.float64)
    B = imgB.astype(np.float64)
    diff = A - B
    return 0.5 * np.sum(diff**2)

def compute_re_ssd(f_img: np.ndarray, m_img: np.ndarray, m2f_img: np.ndarray) -> float:
    """
    Compute Re-SSD(m, f) = SSD(m2f, f) / SSD(m, f).
    """
    denom = ssd(m_img, f_img)
    if denom == 0:
        # If the two images are identical or both constant images, the denominator may be 0
        return float('inf')
    
    numer = ssd(m2f_img, f_img)
    return numer / denom

# -----------------------Compute Re-SSD------------------------------------------------------

# -----------------------ComputeDice---------------------------------------------------------

def dice_score(pred: torch.Tensor,
               target: torch.Tensor,
               eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the Dice coefficient of two binary masks.

    Parameters:
        pred   (torch.Tensor): Predicted mask, shape (1,1,H,W), dtype=torch.bool or {0,1} float/integer
        target (torch.Tensor): Ground-truth mask, shape (1,1,H,W), dtype=torch.bool or {0,1} float/integer
        eps     (float): Small constant to prevent division by zero

    Returns:
        torch.Tensor: Scalar, Dice coefficient in [0,1]
    """
    # Flatten the input to 1D
    pred_flat   = pred.reshape(-1).float()
    target_flat = target.reshape(-1).float()

    # Compute the intersection and each sum
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    # Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    dice = (2.0 * intersection + eps) / (union + eps)

    return dice

# -----------------------ComputeDice---------------------------------------------------------





# -----------------------Compose deformation field (version 1)------------------------------------------------------



def compose_displacements(u1: torch.Tensor,
                          u2: torch.Tensor) -> torch.Tensor:
    """
    Compose two displacement fields u1 and u2: transform by u1 first, then by u2.
    The base_grid generated here has base_grid[0,0,:,:] = row indices (0,1,2,...,H-1)
                     base_grid[0,1,:,:] = column indices (0,1,2,...,W-1)
    The final output is also a displacement field with shape [B,2,H,W].
    """
    B, C, H, W = u1.shape
    assert C == 2, "Only 2D displacement fields are supported"

    # 1) Construct base_grid, shape = [B, 2, H, W]
    ys = torch.arange(0, H, device=u1.device, dtype=u1.dtype)
    xs = torch.arange(0, W, device=u1.device, dtype=u1.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    # After stack, the first channel is grid_y (row index), and the second channel is grid_x (column index)
    base_grid = torch.stack((grid_y, grid_x), dim=0)       # [2, H, W]
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)   # [B, 2, H, W]

    # 2) Perform the first transform first: pts = base_grid + u1
    #    pts is still [B,2,H,W], where
    #      pts[:,0,:,:] = y + u1_y
    #      pts[:,1,:,:] = x + u1_x
    pts = base_grid + u1

    # 3) Convert pts to the [B, H, W, 2] format required by grid_sample
    #    and normalize it: grid_sample requires the last dimension to be (x_norm, y_norm)
    #    where x_norm = 2*(x/(W-1) - 0.5), y_norm = 2*(y/(H-1) - 0.5)
    #    Note that we must "flip" the channel order here
    #    pts_permuted[...,0] is used for x_norm, and pts_permuted[...,1] stores y_norm
    pts = pts.permute(0, 2, 3, 1)  # [B, H, W, 2],  now pts[...,0]=y, pts[...,1]=x

    x = pts[..., 1]
    y = pts[..., 0]
    x_norm = 2.0 * (x / (W - 1) - 0.5)
    y_norm = 2.0 * (y / (H - 1) - 0.5)
    grid_norm = torch.stack((x_norm, y_norm), dim=-1)  # [B, H, W, 2]

    # 4) Perform bilinear interpolation on u2
    u2_warped = F.grid_sample(
        u2,             # [B, 2, H, W]
        grid_norm,      # [B, H, W, 2]
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )  # => [B, 2, H, W]

    # 5) Add them to obtain the composed displacement field
    u_comp = u1 + u2_warped
    return u_comp





def compose_displacements_list(u_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Sequentially compose all displacement fields in a displacement-field list.
    
    Parameters:
      u_list: List of tensors, each of shape [B,2,H,W],represents a series of displacement fields
    Returns:
      u_total: Tensor of shape [B,2,H,W],represents the total displacement field after composing all displacement fields in order
    """
    if not u_list:
        raise ValueError("u_list cannot be empty; at least one displacement field is required")
    
    # Start from the first displacement field
    u_total = u_list[0]
    # Sequentially compose each subsequent displacement field
    for u in u_list[1:]:
        u_total = compose_displacements(u_total, u)
    
    return u_total



# -----------------------Compose deformation field (version 1)------------------------------------------------------


# -----------------------Draw deformation field------------------------------------------------------


def plot_deformation_grid_batch(u_tot, stride=3, batch_idx=0, index=0):
    """
    Draw deformed grid lines in Python/matplotlib (batch supported) and save as jpg:
      - u_tot: np.ndarray or torch.Tensor,shape (B, 2, H, W)
      - stride: int,grid-line sampling stride
      - batch_idx: sample index to draw
      - index: integer used to generate the file name index_field.jpg
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # If it is a torch.Tensor, convert it to numpy (without breaking the original computation graph)
    if hasattr(u_tot, "cpu"):
        u_tot = u_tot.detach().cpu().numpy()

    B, C, H, W = u_tot.shape
    assert 0 <= batch_idx < B, f"batch_idx must be in [0, {B-1}]"
    assert C == 2, "The number of channels must be 2 (x displacement and y displacement)"

    # Extract the displacement field of the specified batch and convert it to (H, W, 2)
    flow = u_tot[batch_idx]               # (2, H, W)
    flow = np.transpose(flow, (1, 2, 0))  # (H, W, 2)

    # -- Base coordinates aligned with SpatialTransformer (pixel centers from 1 to W/H)--
    # Note: this uses 1-based pixel-center coordinates, perfectly matching align_corners=True
    x = np.linspace(1, W, W)      # 1, 2, ..., W
    y = np.linspace(1, H, H)      # 1, 2, ..., H
    x_grid, y_grid = np.meshgrid(x, y, indexing='xy')  # (H, W)

    # Displacement channels: [...,0] is the x direction; [...,1] is the y direction (consistent with the grid_sample convention)
    x_disp = flow[..., 0]
    y_disp = flow[..., 1]

    # Deformed coordinates: total_grid = base_grid + disp
    phyx = x_grid + x_disp
    phyy = y_grid + y_disp

    # Draw grid lines
    dpi = 50
    plt.figure(figsize=(3*W / dpi, 3*H / dpi), dpi=dpi)

    # Row grid (polylines with equal y)
    for row in range(0, H, stride):
        plt.plot(phyx[row, :], phyy[row, :], 'b-', linewidth=3)

    # Column grid (polylines with equal x)
    for col in range(0, W, stride):
        plt.plot(phyx[:, col], phyy[:, col], 'b-', linewidth=3)

    # Canvas settings: keep consistent with 1..W / 1..H coordinates; the y-axis increases from top to bottom, so flip the display
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')  # Equal aspect ratio
    ax.set_xlim(1, W)
    ax.set_ylim(H, 1)  # Equivalent to invert_yaxis, but does not break coordinate semantics

    # Remove axes and margins, fill the canvas
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.xticks([]); plt.yticks([])
    plt.margins(0)
    ax.set_position([0, 0, 1, 1])

    # Save as jpg (assuming field_result_dir exists globally)
    out_path = os.path.join(field_result_dir, f"{index}_field.jpg")
    plt.savefig(out_path, format='jpg', bbox_inches='tight', pad_inches=0)
    plt.close()




# -----------------------Draw deformation field------------------------------------------------------

# -----------------------Draw masks--------------------------------------------------------


def plot_masks(mask1: torch.Tensor,
               mask2: torch.Tensor,
               mask3: torch.Tensor,
               index: int,
               titles=None):
    """
    Save three boolean masks with shape (1,1,H,W) as three EPS images.

    Parameters:
        mask1, mask2, mask3 (torch.Tensor): dtype=torch.bool, shape=(1,1,H,W)
        i (int): Index used to generate the file name
        titles (list of str, optional): Titles of the three images; if None, no title is set
    """
    # Input check
    for idx, m in enumerate((mask1, mask2, mask3), 1):
        if not (isinstance(m, torch.Tensor) and m.dtype == torch.bool):
            raise ValueError(f"mask{idx} must be a Tensor of torch.bool type")
        if m.ndim != 4 or m.shape[0] != 1 or m.shape[1] != 1:
            raise ValueError(f"mask{idx} shape should be (1,1,H,W)")

    # Prepare titles
    if titles is not None:
        if len(titles) != 3:
            raise ValueError("titles length must be 3")

    # Remove batch and channel dimensions and convert to NumPy
    masks_np = [
        mask1.squeeze(0).squeeze(0).cpu().numpy(),
        mask2.squeeze(0).squeeze(0).cpu().numpy(),
        mask3.squeeze(0).squeeze(0).cpu().numpy(),
    ]

    # Used for file names and default titles
    info = [
        ("m_mask",    masks_np[0], titles[0] if titles else None),
        ("m2f_mask",  masks_np[1], titles[1] if titles else None),
        ("f_mask",    masks_np[2], titles[2] if titles else None),
    ]

    # Ensure the output directory exists
    os.makedirs(mask_result_dir, exist_ok=True)

    # Save one by one
    for name, mask_np, title in info:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        out_path = os.path.join(mask_result_dir, f"{name}_{index}.jpg")
        plt.savefig(out_path, format='jpg', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

# -----------------------Draw masks--------------------------------------------------------






def test():
    # Create the required folders and specify the gpu
    
   
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    

    # Read case images
    folder = args.test_dir
    img_example = read_one_image(folder)
    vol_size = img_example.shape[2:]

    
    if args.train_mode:
        UNet_list = []
        for level in range(1, args.cascade_nums + 1):
            model = POTTSNET(args).to(device)
            checkpoint_path = getattr(args, f'supervised_checkpoint{level}_path')  # Dynamically get the path parameter
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            UNet_list.append(model)

        for model in UNet_list:
            model= model.float()  # Ensure model parameters are float type
            model.eval()
        
        # Unpack into independent variables (if later code needs to keep the original variable names)
        UNet1, UNet2, UNet3, UNet4, UNet5, UNet6, UNet7, UNet8, UNet9, UNet10 = UNet_list
    else:
        UNet_list = []
        for level in range(1, args.cascade_nums + 1):
            model = POTTSNET(args).to(device)
            checkpoint_path = getattr(args, f'unsupervised_checkpoint{level}_path')  # Dynamically get the path parameter
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            UNet_list.append(model)

        for model in UNet_list:
            model= model.float()  # Ensure model parameters are float type
            model.eval()

        UNet1, UNet2, UNet3, UNet4, UNet5, UNet6, UNet7, UNet8, UNet9, UNet10 = UNet_list
        

    STN = SpatialTransformer(vol_size).to(device) # Spatial transformer network, used to apply the deformation field to the moving image and generate the registered image, vol_size = [W, H]
    STN = STN.float()
    STN.eval()
    
    
    # opt = Adam(UNet.parameters(), lr=args.lr)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn_1 = losses.second_order_loss
    grad_loss_fn_2 = losses.gradient_loss
    CR_loss_fn = losses.curl_regularizer
    grid_folds = losses.count_grid_folds

   
    # Training loop.
    data_1 = [] # Record Re-SSD
    data_2 = [] # Record Dice
    data_3 = [] # Record MFN
    
    k = 0
    for i in range(1,2): # 1,args.n_iter_in_test + 1
        # Generate the moving images and convert them to tensors.

        # ------------------------------------Test a single image pair------------------------------------------

        # Define the transform pipeline: convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),  # Automatically convert to the [0,1] range, shape (1, H, W) 
        ])

        input_fixed = Image.open(r"./picture_in_SIIMS2020/C_f.png").convert("L")
        input_fixed = transform(input_fixed)  # Convert to tensor and normalize
        input_fixed = input_fixed.unsqueeze(0)
        input_fixed = input_fixed.to(device).float()

        input_moving = Image.open(r"./picture_in_SIIMS2020/C_m.png").convert("L")
        input_moving = transform(input_moving)
        input_moving = input_moving.unsqueeze(0)
        input_moving = input_moving.to(device).float()

        input_moving_temp = input_moving.clone()

        # ------------------------------------Test a single image pair------------------------------------------


        # ------------------------------------Test multiple image pairs------------------------------------------

        # input_fixed = read_one_image(folder) 
        # input_fixed = input_fixed.to(device).float()

        # input_moving = read_one_image(folder)
        # input_moving = input_moving.to(device).float()

        # input_moving_temp = input_moving.clone()

        # ------------------------------------Test multiple image pairs------------------------------------------


        # ------------------------------------Test image pairs with masks(MRI)----------------------------------------

        # mat_dir = r"./data/test_data_with_label"
        # T_list, D_list, T_mask_list, D_mask_list = load_mat_pairs(mat_dir)

        # input_moving = T_list[i % len(T_list)].to(device).float()
        # moving_label = T_mask_list[i % len(T_mask_list)].to(device)  # Use the mask in T_mask_list as the moving-image mask

        # input_fixed = D_list[i % len(D_list)].to(device).float()  # Use the image in D_list as the fixed image
        # fixed_label = D_mask_list[i % len(D_mask_list)].to(device)  # Use the mask in D_mask_list as the fixed-image mask

        # moving_label_64 = moving_label.float()  # Convert the boolean mask to a floating-point mask

        # input_moving_temp = input_moving.clone()

        # ------------------------------------Test image pairs with masks(MRI)----------------------------------------


        # ------------------------------------Test image pairs with masks(ACDC, CAMUS)----------------------------------------

        # h5_path = r"./data/additional_data_with_mask/CAMUS/CAMUS2H_test_100.h5"
        # # r"./data/additional_data_with_mask/CAMUS/CAMUS2H_test_100.h5"
        # # r"./data/additional_data_with_mask/ACDC/acdc_test_54.h5"
        # T_list, D_list, T_mask_list, D_mask_list = load_h5_4d(h5_path)

        # input_moving = T_list[i % len(T_list)].to(device).float()
        # moving_label = T_mask_list[i % len(T_mask_list)].to(device)  # Use the mask in T_mask_list as the moving-image mask

        # input_fixed = D_list[i % len(D_list)].to(device).float()  # Use the image in D_list as the fixed image
        # fixed_label = D_mask_list[i % len(D_mask_list)].to(device)  # Use the mask in D_mask_list as the fixed-image mask

        # moving_label_64 = moving_label.float()  # Convert the boolean mask to a floating-point mask

        # input_moving_temp = input_moving.clone()

        # ------------------------------------Test image pairs with masks(ACDC, CAMUS)----------------------------------------
        
        start=time.time()
       
        flow_list = []  # Used to store the output deformation field of each cascaded UNet
        MFN_list = []  # Total number of deformation-field grid folds

        # ---------------------Registration for the test-data portion of the training data------------------------
        
        for j in range(1, 30): # 1, args.cascade_nums + 1
            UNet = UNet2 # eval(f'UNet{j}')
            flow_m2f = UNet(input_moving, input_fixed)
            MFN = grid_folds(flow_m2f)  # Compute the number of grid folds in the deformation field
            MFN = MFN.float()  # Ensure MFN is a floating-point type
            MFN_list.append(MFN.item())  # Add the MFN of the current cascaded UNet to the total list
            m2f = STN(input_moving, flow_m2f)
            input_moving = m2f  # Update the moving image to the registered image

            # -----------------Apply the deformation field to the moving-image mask; for image pairs without masks, this step needs to be commented out------------
            
            # m2f_label = STN(moving_label_64, flow_m2f)
            # moving_label_64 = m2f_label  # Update the moving-image mask  

            # -----------------Apply the deformation field to the moving-image mask; for image pairs without masks, this step needs to be commented out------------

            flow_list.append(flow_m2f)
        
        
        # ---------------------Registration for the test-data portion of the training data------------------------

        # ---------------------Registration for untrained data------------------------

        # # --------- Preparation in advance ---------
        # threshold = 0.001               # You can change it to parameter args.threshold
        # max_cascades = 30               # or args.cascade_nums
        # prev_best_re_ssd = float('inf') # Best value of the previous level, initially set to +inf

        # flow_list = []
        # MFN_list  = []
        # re_ssd_list = []                # Optional, if the best re_ssd for each level needs to be recorded

        # # --------- Cascade loop ---------
        # for lvl in range(max_cascades):

        #     # ------ 1. Select the best UNet in this level ------
        #     best_re_ssd = float('inf')
        #     best_flow   = None
        #     best_m2f    = None
        #     best_idx    = None          # Record the UNet index (0-based)
        #     best_dice   = 0

        #     for idx, UNet in enumerate(UNet_list, start=1):  # idx Starting from 1 is more intuitive
        #         flow_m2f = UNet(input_moving, input_fixed)         # (B,2,H,W)
        #         m2f = STN(input_moving, flow_m2f)          # (B,1,H,W)

        #         m2f_label = STN(moving_label_64, flow_m2f)
        #         m2f_label_temp = (m2f_label > 0.5)


        #         re_ssd = compute_re_ssd(input_fixed[0, 0, ...].cpu().detach().numpy(),
        #                        input_moving_temp[0, 0, ...].cpu().detach().numpy(),
        #                        m2f[0, 0, ...].cpu().detach().numpy())
        #         dicescore = dice_score(m2f_label_temp, fixed_label)

        #         if dicescore > best_dice:        # Record the current best
        #             best_re_ssd = re_ssd
        #             best_dice   = dicescore
        #             best_flow   = flow_m2f
        #             best_m2f    = m2f
        #             best_idx    = idx           # 1-based is convenient for viewing

        #     # ------ 2. Update state and save statistics ------
        #     MFN = grid_folds(best_flow).float()    # Fold-count statistics
        #     MFN_list.append(MFN.item())
        #     flow_list.append(best_flow)            # Save the best flow
        #     re_ssd_list.append(best_re_ssd)

        #     # Print or log (optional)
        #     print(f"[Level {lvl+1}] Pick UNet{best_idx}: dice={best_dice:.6f}, MFN={MFN.item():.1f}")

        #     # Update the moving image to prepare for the next level
        #     input_moving = best_m2f
        #     # -----------------Apply the deformation field to the moving-image mask; for image pairs without masks, this step needs to be commented out------------
            
            
        #     moving_label_64 = m2f_label  # Update the moving-image mask  

        #     # -----------------Apply the deformation field to the moving-image mask; for image pairs without masks, this step needs to be commented out------------

        #     # ------ 3. Convergence check ------
        #     if abs(prev_best_re_ssd - best_re_ssd) < threshold:
        #         print(f"Converged (Δre_ssd < {threshold}). Early stop at level {lvl+1}.")
        #         break
        #     prev_best_re_ssd = best_re_ssd

        
        # ---------------------Registration for untrained data------------------------

        flow_list.reverse()  # Reverse the order to compose deformation fields from the innermost layer to the outermost layer
        flow_m2f = compose_displacements_list(flow_list) # compose_n_flows

        #----------------------Convert to boolean tensor; image pairs without masks need to comment this out-----------------------------
        # m2f_label = (m2f_label > 0.5) 
        #----------------------Convert to boolean tensor; image pairs without masks need to comment this out-----------------------------


        # Calculate loss
        sim_loss = sim_loss_fn(m2f, input_fixed)
        primary_sim = sim_loss_fn(input_moving_temp, input_fixed)  # Similarity loss between the moving image and the fixed image
        
        grad_loss = grad_loss_fn_1(flow_m2f) + grad_loss_fn_2(flow_m2f) + CR_loss_fn(flow_m2f)
        loss = sim_loss + args.alpha_1 * grad_loss
        MFN = grid_folds(flow_m2f)
        
        #-------------------Compute dice; image pairs without masks need to comment this out----------------------------
        # dicescore = dice_score(m2f_label, fixed_label)
        #-------------------Compute dice; image pairs without masks need to comment this out----------------------------

        # Calculate Re-SSD
        re_ssd = compute_re_ssd(input_fixed[0, 0, ...].cpu().detach().numpy(),
                               input_moving_temp[0, 0, ...].cpu().detach().numpy(),
                               m2f[0, 0, ...].cpu().detach().numpy()) #[0, 0, ...] is used to extract the image with batch_size=1
        
        end=time.time()
        MFN = sum(MFN_list)

        tt=end-start
        # print("i: %d  loss: %f  sim: %f  primary_sim: %f  grad: %f  validation_time: %f  re_ssd: %f  MFN: %d  dice: %f"  % (i, loss.item(), sim_loss.item(),primary_sim.item(),grad_loss.item(), tt, re_ssd, MFN, dicescore), flush=True)
        print("i: %d  loss: %f  sim: %f  primary_sim: %f  grad: %f  validation_time: %f  re_ssd: %f  MFN: %d"  % (i, loss.item(), sim_loss.item(),primary_sim.item(),grad_loss.item(), tt, re_ssd, MFN), flush=True)

        
        
        if re_ssd != 'inf':
            data_1.append(re_ssd.item()) 
            # data_2.append(dicescore.item())
            data_3.append(MFN)  # Add the MFN of the current cascaded UNet to the total list
            
            # Save images
            m_name = str(i) + "_m"
            m2f_name = str(i) + "_m2f"
            f_name = str(i) + "_f"
            save_image(input_fixed, f_name)
            save_image(input_moving_temp, m_name)
            save_image(m2f, m2f_name)
            plot_deformation_grid_batch(flow_m2f, stride=3, batch_idx=0, index=i)  # Draw the deformation-field grid
            #-------------------Draw masks; image pairs without masks need to comment this out----------------------------
            # plot_masks(moving_label, m2f_label, fixed_label,index=i,
            #         titles=["Moving Image Mask", "m2f Image Mask", "Fixed Image Mask"])
            #-------------------Draw masks; image pairs without masks need to comment this out----------------------------
            print("warped images have saved.")
            k += 1

    print("total number of saved images: ", k)    
    

    average_re_ssd = np.mean(data_1)
    average_dice = np.mean(data_2)
    average_MFN = np.mean(data_3)

    std_re_ssd = np.std(data_1, ddof=1)
    std_dice = np.std(data_2, ddof=1)
    std_MFN = np.std(data_3, ddof=1)

    print(f"\nAverage Re-SSD of all groups = {average_re_ssd:.6f}")
    print(f"\nAverage Dice of all groups = {average_dice:.6f}")
    print(f"\nAverage MFN of all groups = {average_MFN:.6f}")

    print(f"\nRe-SSD standard deviation of all groups = {std_re_ssd:.6f}")
    print(f"\nDice standard deviation of all groups = {std_dice:.6f}")
    print(f"\nMFN standard deviation of all groups = {std_MFN:.6f}")

    
    

    

    
    
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    test()
