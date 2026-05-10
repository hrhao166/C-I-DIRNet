# python imports
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import random
import warnings
import time
import h5py
# external imports
import torch
import numpy as np
import scipy.io as sio
import scipy.io
# import SimpleITK as sitk
from PIL import Image
from torch.optim import Adam
import torch.utils.data as Data
from torchvision import transforms
# internal imports
from model import losses
from model.config import args
from model.PottsMorph_model import POTTSNET, SpatialTransformer
from generator import supervised_generator, unsupervised_generator


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params





def read_one_image(folder_path): # Read image pairs for unsupervised learning
    """
    Randomly read a PNG image from the specified folder and return an image tensor normalized to the [0,1] range.

    Parameters:
        folder_path (str): Folder path for storing PNG images

    Returns:
        torch.Tensor: Normalized image tensor, shape (1, H, W), value range [0,1]

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



def load_following_mat_files(folder_path): # Function shared by supervised and unsupervised learning, reads m, f, and m2f
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




def supervised_train(cascade_level, data_folder_path, alpha_2):
    # Create the required folders and specify the gpu
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    
    while True:
        print(f"\n=== A new training round starts: alpha_2 = {alpha_2:.2f} ===")
        # Read the example image
        folder = r"D:\PottsMorph_data\bg_150.h5"
        T_list, D_list, label_field_list = load_h5_data_shuffled(folder)

        input_img = T_list[0]  # Take the first moving image
        vol_size = input_img.shape[2:] # [H, W]
    
    
        UNet = POTTSNET(args).to(device)
        STN = SpatialTransformer(vol_size).to(device) # Spatial transformer network, used to apply the deformation field to the moving image and generate the registered image
        
    
        UNet.train()
        STN.train()
        # Number of model parameters
        print("UNet: ", count_parameters(UNet))
        print("STN: ", count_parameters(STN))

        # Set optimizer and losses
        opt = Adam(UNet.parameters(), lr=args.lr)
        sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
        grad_loss_fn_1 = losses.second_order_loss
        grad_loss_fn_2 = losses.gradient_loss
        field_sim_loss = losses.mse_diff
        grid_folds = losses.count_grid_folds

        if cascade_level == 1:
            T_list, D_list, label_field_list = load_h5_data_shuffled(data_folder_path)
        else:
            T_list, D_list, m2f_list = load_following_mat_files(data_folder_path) # m_list, f_list, m2f_list

        # Used to accumulate MFN over the whole training loop
        sum_MFN = 0

        # Training loop.
        for i in range(1, args.n_iter + 1): # 1, args.n_iter + 1
            # Generate the moving images and convert them to tensors.
            if cascade_level == 1:
            
                input_fixed = D_list[(i-1) % len(D_list)]
                input_fixed = input_fixed.to(device).float()

                input_moving = T_list[(i-1) % len(T_list)]
                input_moving = input_moving.to(device).float()

                label_field = label_field_list[i % len(label_field_list)]  # Label field
                label_field = label_field.to(device).float()

                start=time.time()
                # Run the data through the model to produce warp and flow field
                flow_m2f = UNet(input_moving, input_fixed)
                m2f = STN(input_moving, flow_m2f)
                
                # Calculate loss
                MFN = grid_folds(flow_m2f)

                if i > 1000:
                    sum_MFN += MFN
                
                figure_sim = sim_loss_fn(m2f, input_fixed)  # Similarity loss between the registered image and the fixed image
                primary_sim = sim_loss_fn(input_moving, input_fixed)  # Similarity loss between the moving image and the fixed image
        
                field_loss = field_sim_loss(flow_m2f, label_field)  # Compute the similarity loss between the flow field and the label field
                regularization_loss = args.alpha_1 * (grad_loss_fn_1(flow_m2f) + grad_loss_fn_2(flow_m2f)) 
                loss = args.lambda_1 * figure_sim + args.lambda_2 * field_loss + regularization_loss
        
                end=time.time()
                tt=end-start
                print("i: %d  loss: %f  field_loss: %f  reg_loss: %f  train_time: %f  co_sim: %f  primary_sim: %f  MFN: %d"  % (i, loss.item(), field_loss.item(), regularization_loss.item(), tt, figure_sim.item(),primary_sim.item(),MFN), flush=True)

                # Backwards and optimize
                opt.zero_grad()
                loss.backward()
                opt.step()
                
            else:
            
                input_fixed = D_list[(i-1) % len(D_list)]
                input_fixed = input_fixed.to(device).float()

                input_moving = m2f_list[(i-1) % len(T_list)]
                input_moving = input_moving.to(device).float()

                start=time.time()
                # Run the data through the model to produce warp and flow field
                flow_m2f = UNet(input_moving, input_fixed)
                m2f = STN(input_moving, flow_m2f)

                # Calculate loss
                MFN = grid_folds(flow_m2f)

                if i > 1000:
                    sum_MFN += MFN

                figure_sim = sim_loss_fn(m2f, input_fixed)  # Similarity loss between the registered image and the fixed image
                primary_sim = sim_loss_fn(input_moving, input_fixed)  # Similarity loss between the moving image and the fixed image

                regularization_loss = args.alpha_1 * (grad_loss_fn_1(flow_m2f) + grad_loss_fn_2(flow_m2f)) 
                
                loss = args.lambda_1 * figure_sim + regularization_loss

                end=time.time()
                tt=end-start
                print("i: %d  loss: %f  reg_loss: %f  train_time: %f  co_sim: %f  primary_sim: %f  MFN: %d"  % (i, loss.item(), regularization_loss.item(), tt, figure_sim.item(),primary_sim.item(),MFN), flush=True)

                # Backwards and optimize
                opt.zero_grad()
                loss.backward()
                opt.step()

                
        # After training ends, make a decision based on sum_MFN
        print(f"\nOne training round ended, accumulated sum_MFN = {sum_MFN}")
        if sum_MFN < 0:
            alpha_2 -= 0.3
            print(f"sum_MFN < 0,adjust alpha_2 to {alpha_2:.2f},restart training.")
            continue # Jump directly back to the top of while: the model/optimizer have both been rebuilt
        elif sum_MFN > 1500000:
            alpha_2 += 0.3
            print(f"sum_MFN > 1500000,adjust alpha_2 to {alpha_2:.2f},restart training.")
            continue # Same as above, restart again with the new alpha_2
        else:
            save_file_name = '150_no_regularization_supervised_cascade{}_{}.pth'.format(cascade_level, i)  # First level of the supervised cascade network
            save_file_path = os.path.join(args.supervised_model_dir, save_file_name)
            torch.save(UNet.state_dict(), save_file_path)
            print(f"Training complete, sum_MFN={sum_MFN} is within the [0,1500000] range; model saved to {save_file_path}", flush=True)
            break # Condition satisfied; exit while and save the model


          
    
def unsupervised_train(cascade_level, data_folder_path, alpha_2):
    # Create the required folders and specify the gpu
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    while True:
        print(f"\n=== A new training round starts: alpha_2 = {alpha_2:.2f} ===")
        # Read the example image
        folder = r"D:\PottsMorph_data\bg_10_double.h5"
        T_list, D_list, label_field_list = load_h5_data_shuffled(folder)

        input_img = T_list[0]  # Take the first moving image
        vol_size = input_img.shape[2:] # [H, W]
        
        
        UNet = POTTSNET(args).to(device)
        STN = SpatialTransformer(vol_size).to(device) # Spatial transformer network, used to apply the deformation field to the moving image and generate the registered image
        
        
        UNet.train()
        STN.train()
        # Number of model parameters
        print("UNet: ", count_parameters(UNet))
        print("STN: ", count_parameters(STN))

        # Set optimizer and losses
        opt = Adam(UNet.parameters(), lr=args.lr)
        sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
        grad_loss_fn_1 = losses.second_order_loss
        grad_loss_fn_2 = losses.gradient_loss
        CR_loss_fn = losses.curl_regularizer
        grid_folds = losses.count_grid_folds

        if cascade_level == 1:
            T_list, D_list, label_field_list = load_h5_data_shuffled(data_folder_path)
        else:
            T_list, D_list, m2f_list = load_following_mat_files(data_folder_path) # m_list, f_list, m2f_list
        
        # Used to accumulate MFN over the whole training loop
        sum_MFN = 0

        # Training loop.
        for i in range(1, args.n_iter + 1): # 1, args.n_iter + 1
            # Generate the moving images and convert them to tensors.
            if cascade_level == 1:
                input_fixed = D_list[(i-1) % len(D_list)]
                input_fixed = input_fixed.to(device).float()

                input_moving = T_list[(i-1) % len(T_list)]
                input_moving = input_moving.to(device).float()

                # Unsupervised training does not need to read the label deformation field

                

                start=time.time()
                # Run the data through the model to produce warp and flow field
                flow_m2f = UNet(input_moving, input_fixed)
                m2f = STN(input_moving, flow_m2f)
                
                # Calculate loss
                MFN = grid_folds(flow_m2f)

                if i > 1000:
                    sum_MFN += MFN
                
                figure_sim = sim_loss_fn(m2f, input_fixed)  # Similarity loss between the registered image and the fixed image
                primary_sim = sim_loss_fn(input_moving, input_fixed)  # Similarity loss between the moving image and the fixed image
            
                regularization_loss = args.alpha_1 * (grad_loss_fn_1(flow_m2f) + grad_loss_fn_2(flow_m2f)) 
                
                loss = args.lambda_1 * figure_sim + regularization_loss
            
                end=time.time()
                tt=end-start
                print("i: %d  loss: %f  reg_loss: %f  train_time: %f  co_sim: %f  primary_sim: %f  MFN: %d"  % (i, loss.item(), regularization_loss.item(), tt, figure_sim.item(),primary_sim.item(),MFN), flush=True)

                # Backwards and optimize
                opt.zero_grad()
                loss.backward()
                opt.step()


            else:
                input_fixed = D_list[(i-1) % len(D_list)]
                input_fixed = input_fixed.to(device).float()

                input_moving = m2f_list[(i-1) % len(T_list)]
                input_moving = input_moving.to(device).float()

                start=time.time()
                # Run the data through the model to produce warp and flow field
                flow_m2f = UNet(input_moving, input_fixed)
                m2f = STN(input_moving, flow_m2f)

                # Calculate loss
                MFN = grid_folds(flow_m2f)

                if i > 1000:
                    sum_MFN += MFN
                
                figure_sim = sim_loss_fn(m2f, input_fixed)  # Similarity loss between the registered image and the fixed image
                primary_sim = sim_loss_fn(input_moving, input_fixed)  # Similarity loss between the moving image and the fixed image

                regularization_loss = args.alpha_1 * (grad_loss_fn_1(flow_m2f) + grad_loss_fn_2(flow_m2f)) 
                loss = args.lambda_1 * figure_sim + regularization_loss

                end=time.time()
                tt=end-start
                print("i: %d  loss: %f  reg_loss: %f  train_time: %f  co_sim: %f  primary_sim: %f  MFN: %d"  % (i, loss.item(), regularization_loss.item(), tt, figure_sim.item(),primary_sim.item(),MFN), flush=True)

                # Backwards and optimize
                opt.zero_grad()
                loss.backward()
                opt.step()


        # After training ends, make a decision based on sum_MFN
        print(f"\nOne training round ended, accumulated sum_MFN = {sum_MFN}")
        if sum_MFN < 0:
            alpha_2 -= 0.2
            print(f"sum_MFN < 0,adjust alpha_2 to {alpha_2:.2f},restart training.")
            continue # Jump directly back to the top of while: the model/optimizer have both been rebuilt
        elif sum_MFN > 30000000:
            alpha_2 += 0.3
            print(f"sum_MFN > 30,adjust alpha_2 to {alpha_2:.2f},restart training.")
            continue # Same as above, restart again with the new alpha_2
        else:
            # 0 <= sum_MFN <= 30, training ends
            save_file_name = '10_256_unsupervised_cascade{}_{}.pth'.format(cascade_level, i)  # First level of the unsupervised cascade network
            save_file_path = os.path.join(args.unsupervised_model_dir, save_file_name)
            torch.save(UNet.state_dict(), save_file_path)
            print(f"Training complete, sum_MFN={sum_MFN} is within the [0,30] range; model saved to {save_file_path}", flush=True)
            break # Condition satisfied; exit while and save the model


            



if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    if args.train_mode:
        # Supervised training mode
        print("Starting supervised training...")
        
        for i in range(1, args.cascade_nums + 1):
            if i == 1:
                # First level of the cascade network
                print("Training supervised cascade level 1...")
                supervised_train(cascade_level=1, data_folder_path=args.supervised_train_dir, alpha_2=args.supervised_alpha_2[0])
                print("Generating supervised data for cascade level 1...")
                supervised_generator(cascade_level=1, data_folder_path=args.supervised_train_dir, alpha_2=args.supervised_alpha_2[0])

            else:
                # Subsequent levels of the cascade network
                print(f"Training supervised cascade level {i}...")
                supervised_train(cascade_level=i, data_folder_path=args.supervised_generation_dir, alpha_2=args.supervised_alpha_2[i-1])
                print(f"Generating supervised data for cascade level {i}...")
                supervised_generator(cascade_level=i, data_folder_path=args.supervised_generation_dir, alpha_2=args.supervised_alpha_2[i-1])
            
    else:
        # Unsupervised training mode
        print("Starting unsupervised training...")
        
        for i in range(1, args.cascade_nums + 1):
            if i == 1:
                # First level of the cascade network
                print("Training unsupervised cascade level 1...")
                unsupervised_train(cascade_level=1, data_folder_path=args.unsupervised_train_dir, alpha_2=args.unsupervised_alpha_2[0])
                print("Generating unsupervised data for cascade level 1...")
                unsupervised_generator(cascade_level=1, data_folder_path=args.unsupervised_train_dir, alpha_2=args.unsupervised_alpha_2[0])
            else:
                # Subsequent levels of the cascade network
                print(f"Training unsupervised cascade level {i}...")
                unsupervised_train(cascade_level=i, data_folder_path=args.unsupervised_generation_dir, alpha_2=args.unsupervised_alpha_2[i-1])
                print(f"Generating unsupervised data for cascade level {i}...")
                unsupervised_generator(cascade_level=i, data_folder_path=args.unsupervised_generation_dir, alpha_2=args.unsupervised_alpha_2[i-1])

    print("Training completed.")

            
    
    
