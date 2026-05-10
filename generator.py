# python imports
import os
import glob
import h5py
import random
import warnings
import time
import scipy.io
import scipy.io as sio
# external imports
import torch
import numpy as np
# import SimpleITK as sitk
from PIL import Image
from torchvision import transforms
import torch.utils.data as Data
# internal imports
from model import losses
from model.config import args
from model.datagenerators import Dataset
from model.PottsMorph_model import POTTSNET, SpatialTransformer


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params



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



def load_h5_data_shuffled(h5_path: str, seed: int = None):
    """
    Read the T, D, and transfield datasets from the specified .h5 file, split batches, and finally randomly shuffle the sample order,
    Returns:
      - T_list:            List[torch.FloatTensor], each shape (1,1,128,128)
      - D_list:            List[torch.FloatTensor], each shape (1,1,128,128)
      - label_field_list:  List[torch.FloatTensor], each shape (1,2,128,128)
    Elements under the same index still correspond to the same batch sample; only the overall order is shuffled.
    
    Parameters:
      - h5_path: HDF5 file path
      - seed:    Random seed (optional); the same seed keeps each shuffle order consistent
    """
    # Define the T/D transform
    transform = transforms.Compose([
        transforms.ToTensor(),  # HxW -> 1xHxW, and normalize to [0,1]
    ])
    
    T_list = []
    D_list = []
    label_field_list = []
    
    # 1) Read in and split
    with h5py.File(h5_path, 'r') as f:
        T_ds  = f['T']           # (B,1,128,128)
        D_ds  = f['D']           # (B,1,128,128)
        TF_ds = f['transfield']  # (B,2,128,128)
        B = T_ds.shape[0]
        
        for i in range(B):
            # Process T
            T_np = T_ds[i, 0, :, :]
            T_tensor = transform(T_np).unsqueeze(0).float()  # -> (1,1,128,128)
            # Process D
            D_np = D_ds[i, 0, :, :]
            D_tensor = transform(D_np).unsqueeze(0).float()  # -> (1,1,128,128)
            # Process transfield
            TF_np = TF_ds[i, :, :, :]
            TF_tensor = torch.from_numpy(TF_np.astype('float32')).unsqueeze(0)  # -> (1,2,128,128)
            
            T_list.append(T_tensor)
            D_list.append(D_tensor)
            label_field_list.append(TF_tensor)
    
    # 2) Generate shuffled indices
    indices = list(range(len(T_list)))
    if seed is not None:
        random.seed(seed)
    random.shuffle(indices)
    
    # 3) Reorder the three lists according to the shuffled indices
    T_list_shuffled          = [T_list[i] for i in indices]
    D_list_shuffled          = [D_list[i] for i in indices]
    label_field_list_shuffled= [label_field_list[i] for i in indices]
    
    return T_list_shuffled, D_list_shuffled, label_field_list_shuffled



def load_following_mat_files(folder_path):
    """
    Load .mat files and return m, f, and m2f as torch tensors
    
    Parameters:
        folder_path (str): Folder path containing .mat files
        
    Returns:
        tuple: Three lists (m, f, m2f), where each element is a torch.Tensor with shape (1,1,H,W)
    """
    # Get all .mat files
    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    
    # Shuffle file order
    random.shuffle(mat_files)
    
    # Initialize empty lists to store data
    m_list, f_list, m2f_list = [], [], []
    
    for mat_file in mat_files:
        file_path = os.path.join(folder_path, mat_file)
        data = scipy.io.loadmat(file_path)
        
        # Extract T, D, and m2f variables and reshape them to (1,1,H,W)
        m = data['m']
        f = data['f']
        m2f = data['m2f']
        
        # Reshape to (1,1,H,W) and convert to torch.float32 type
        m = torch.tensor(m, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        f = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        m2f = torch.tensor(m2f, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        m_list.append(m)
        f_list.append(f)
        m2f_list.append(m2f)
    
    return m_list, f_list, m2f_list


def save_multi_image_mat(img_m, img_f, img_m2f, name, save_dir):
    """
    Save three medical images as different variables in the same MAT file
    
    :param img_m: Input image m (Tensor), shape (1, 1, H, W)
    :param img_f: Input image f (Tensor), shape (1, 1, H, W)
    :param img_m2f: Input image m2f (Tensor), shape (1, 1, H, W)
    :param name: Saved file name (without extension)
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare the dictionary to save
    save_dict = {}
    
    # Process and add each image to the dictionary
    for img, var_name in zip([img_m, img_f, img_m2f], ["m", "f", "m2f"]):
        # Convert to a NumPy array and remove batch and channel dimensions
        img_np = img[0, 0, ...].cpu().detach().numpy()
        
        # Add to the save dictionary
        save_dict[var_name] = img_np
    
    # Prepare the save path
    save_path = os.path.join(save_dir, f"{name}.mat")
    
    # Save as MAT file
    sio.savemat(save_path, save_dict)



def supervised_generator(cascade_level, data_folder_path, alpha_2):
    # Create the required folders and specify the gpu
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    if cascade_level == 1:  
        m_list, f_list, label_field_list = load_h5_data_shuffled(data_folder_path)  # Read the mat list
    else:  # When the cascade network level is greater than 1, use the saved mat file list
        m_list, f_list, m2f_list = load_following_mat_files(data_folder_path)  # Read the mat list

    
    img_example = f_list[0]  # Take the first image as an example
    vol_size = img_example.shape[2:] # [H, W]
    
    
    UNet = POTTSNET(args).to(device)
    checkpoint_attr = f"supervised_checkpoint{cascade_level}_path"
    checkpoint_path = getattr(args, checkpoint_attr)
    UNet.load_state_dict(torch.load(checkpoint_path))

    STN = SpatialTransformer(vol_size).to(device) # Spatial transformer network, used to apply the deformation field to the moving image and generate the registered image, vol_size = [D, W, H]
   
    UNet.eval()
    STN.eval()
    # Number of model parameters
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))

    # Set losses
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn_1 = losses.second_order_loss
    grad_loss_fn_2 = losses.gradient_loss
    grid_folds = losses.count_grid_folds

    
    # generation loop.
    for i in range(1, args.n_iter + 1): # 1, args.n_iter + 1
        # Generate the moving images and convert them to tensors.
        if cascade_level == 1:
            input_moving = m_list[i % len(m_list)]  # Use modulo arithmetic to keep the index within range
            input_fixed = f_list[i % len(f_list)]

            input_moving = input_moving.to(device).float()
            input_fixed = input_fixed.to(device).float()

            input_moving_primary = input_moving.clone() # primary moving image

            start=time.time()
            # Run the data through the model to produce warp and flow field
            flow_m2f = UNet(input_moving, input_fixed)
            m2f = STN(input_moving, flow_m2f)

            # Calculate loss
            MFN = grid_folds(flow_m2f)
            figure_sim = sim_loss_fn(m2f, input_fixed)  # Similarity loss between the registered image and the fixed image
            primary_sim = sim_loss_fn(input_moving_primary, input_fixed)  # Similarity loss between the moving image and the fixed image
            regularization_loss = args.alpha_1 * (grad_loss_fn_1(flow_m2f) + grad_loss_fn_2(flow_m2f))
            loss = args.lambda_1 * figure_sim + regularization_loss

            re_ssd = compute_re_ssd(input_fixed[0, 0, ...].cpu().detach().numpy(),
                               input_moving_primary[0, 0, ...].cpu().detach().numpy(),
                               m2f[0, 0, ...].cpu().detach().numpy()) #[0, 0, ...] is used to extract the image with batch_size=1
       
            end=time.time()
            tt=end-start

            print("i: %d  loss: %f  sim: %f  primary_sim: %f  validation_time: %f  re_ssd: %f  MFN: %d"  % (i, loss.item(), figure_sim.item(),primary_sim.item(), tt, re_ssd, MFN), flush=True)
        

            if i % args.n_save_iter_in_generation == 0:
                image_pair_name = str(i) + "_pair"
                save_multi_image_mat(input_moving_primary, input_fixed, m2f, image_pair_name, args.supervised_generation_dir) # m, f, m2f
            
            
        else:  # When the cascade network level is greater than 1, use the saved mat file list
            input_moving = m2f_list[i % len(m2f_list)]  # Use modulo arithmetic to keep the index within range
            input_fixed = f_list[i % len(f_list)]

            input_moving = input_moving.to(device).float()
            input_fixed = input_fixed.to(device).float()

            input_moving_primary = m_list[i % len(m_list)].clone() # primary moving image
            input_moving_primary = input_moving_primary.to(device).float()

            start=time.time()
            # Run the data through the model to produce warp and flow field
            flow_m2f = UNet(input_moving, input_fixed)
            m2f = STN(input_moving, flow_m2f)

            # Calculate loss
            MFN = grid_folds(flow_m2f)
            figure_sim = sim_loss_fn(m2f, input_fixed)  # Similarity loss between the registered image and the fixed image
            primary_sim = sim_loss_fn(input_moving_primary, input_fixed)  # Similarity loss between the moving image and the fixed image
            regularization_loss = args.alpha_1 * (grad_loss_fn_1(flow_m2f) + grad_loss_fn_2(flow_m2f))
            loss = args.lambda_1 * figure_sim + regularization_loss

            re_ssd = compute_re_ssd(input_fixed[0, 0, ...].cpu().detach().numpy(),
                               input_moving_primary[0, 0, ...].cpu().detach().numpy(),
                               m2f[0, 0, ...].cpu().detach().numpy()) #[0, 0, ...] is used to extract the image with batch_size=1
       
            end=time.time()
            tt=end-start

            print("i: %d  loss: %f  sim: %f  primary_sim: %f  validation_time: %f  re_ssd: %f  MFN: %d"  % (i, loss.item(), figure_sim.item(),primary_sim.item(), tt, re_ssd, MFN), flush=True)
        

            if i % args.n_save_iter_in_generation == 0 and MFN <= 3:
                image_pair_name = str(i) + "_pair"
                save_multi_image_mat(input_moving_primary, input_fixed, m2f, image_pair_name, args.supervised_generation_dir) # m, f, m2f


  


def unsupervised_generator(cascade_level, data_folder_path, alpha_2):
    # Create the required folders and specify the gpu
    
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    if cascade_level == 1:  
        m_list, f_list, label_field_list = load_h5_data_shuffled(data_folder_path)  # Read the mat list
    else:  # When the cascade network level is greater than 1, use the saved mat file list
        m_list, f_list, m2f_list = load_following_mat_files(data_folder_path)  # Read the mat list
    

    img_example = f_list[0]  # Take the first image as an example
    vol_size = img_example.shape[2:]
    # [H, W]


    UNet = POTTSNET(args).to(device)
    
    checkpoint_attr = f'unsupervised_checkpoint{cascade_level}_path' # Build the attribute name for the checkpoint path (for example: cascaded_level=2 -> 'checkpoint2_path')
    if not hasattr(args, checkpoint_attr):
        raise ValueError(f"Invalid cascaded_level: {cascade_level} - No corresponding checkpoint path")
    checkpoint_path = getattr(args, checkpoint_attr)
    UNet.load_state_dict(torch.load(checkpoint_path))

    STN = SpatialTransformer(vol_size).to(device) # Spatial transformer network, used to apply the deformation field to the moving image and generate the registered image, vol_size = [D, W, H]
    
    UNet.eval()
    STN.eval()
    # Number of model parameters
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))

    # Set losses
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn_1 = losses.second_order_loss
    grad_loss_fn_2 = losses.gradient_loss
    CR_loss_fn = losses.curl_regularizer
    grid_folds = losses.count_grid_folds

    

    # Training loop.
    for i in range(1, args.n_iter + 1): 
        # Generate the moving images and convert them to tensors.

        if cascade_level == 1: # When the cascade network level is 1, directly read random PNG images from the specified directory
            input_moving = m_list[i % len(m_list)]  # Use modulo arithmetic to keep the index within range
            input_fixed = f_list[i % len(f_list)]

            input_moving = input_moving.to(device).float()
            input_fixed = input_fixed.to(device).float()

            input_moving_primary = input_moving.clone() # primary moving image

            start=time.time()

            # Run the data through the model to produce warp and flow field
            flow_m2f = UNet(input_moving, input_fixed)
            m2f = STN(input_moving, flow_m2f)

            # Calculate loss
            MFN = grid_folds(flow_m2f)
            figure_sim = sim_loss_fn(m2f, input_fixed)  # Similarity loss between the registered image and the fixed image
            primary_sim = sim_loss_fn(input_moving_primary, input_fixed)  # Similarity loss between the moving image and the fixed image
            regularization_loss = args.alpha_1 * (grad_loss_fn_1(flow_m2f) + grad_loss_fn_2(flow_m2f)) + alpha_2 * CR_loss_fn(flow_m2f)
            loss = args.lambda_1 * figure_sim + regularization_loss

            re_ssd = compute_re_ssd(input_fixed[0, 0, ...].cpu().detach().numpy(),
                                    input_moving_primary[0, 0, ...].cpu().detach().numpy(),
                                    m2f[0, 0, ...].cpu().detach().numpy()) #[0, 0, ...] is used to extract the image with batch_size=1 
       
            end=time.time()
            tt=end-start
        
            print("i: %d  loss: %f  sim: %f  primary_sim: %f  validation_time: %f  re_ssd: %f  MFN: %d"  % (i, loss.item(), figure_sim.item(),primary_sim.item(), tt, re_ssd, MFN), flush=True)

            if i % args.n_save_iter_in_generation == 0:
                # Save images
                image_pair_name = str(i) + "_pair"
                save_multi_image_mat(input_moving_primary, input_fixed, m2f, image_pair_name, args.unsupervised_generation_dir) # m, f, m2f
        
        else:  # When the cascade network level is greater than 1, use the saved mat file list 

            m_tensor = m_list[(i-1) % len(m_list)]  # Use modulo arithmetic to keep the index within range
            input_fixed = f_list[(i-1) % len(f_list)]  
            input_moving = m2f_list[(i-1) % len(m2f_list)]

            input_fixed = input_fixed.to(device).float()  
            input_moving = input_moving.to(device).float()
            m_tensor = m_tensor.to(device).float()

            input_moving_primary = m_tensor.clone() # primary moving image

            start=time.time()

            # Run the data through the model to produce warp and flow field
            flow_m2f = UNet(input_moving, input_fixed)
            m2f = STN(input_moving, flow_m2f)

            # Calculate loss
            MFN = grid_folds(flow_m2f)
            figure_sim = sim_loss_fn(m2f, input_fixed)  # Similarity loss between the registered image and the fixed image
            primary_sim = sim_loss_fn(input_moving_primary, input_fixed)  # Similarity loss between the moving image and the fixed image
            regularization_loss = args.alpha_1 * (grad_loss_fn_1(flow_m2f) + grad_loss_fn_2(flow_m2f)) + alpha_2 * CR_loss_fn(flow_m2f)
            loss = args.lambda_1 * figure_sim + regularization_loss

            re_ssd = compute_re_ssd(input_fixed[0, 0, ...].cpu().detach().numpy(),
                                    input_moving_primary[0, 0, ...].cpu().detach().numpy(),
                                    m2f[0, 0, ...].cpu().detach().numpy()) #[0, 0, ...] is used to extract the image with batch_size=1 
       
            end=time.time()
            tt=end-start
        
            print("i: %d  loss: %f  sim: %f  primary_sim: %f  validation_time: %f  re_ssd: %f  MFN: %d"  % (i, loss.item(), figure_sim.item(),primary_sim.item(), tt, re_ssd, MFN), flush=True)

            if i % args.n_save_iter_in_generation == 0:
                # Save images
                image_pair_name = str(i) + "_pair"
                save_multi_image_mat(input_moving_primary, input_fixed, m2f, image_pair_name, args.unsupervised_generation_dir) # m, f, m2f
            
            

        
        
        

        

        

        

        
            
            
    
