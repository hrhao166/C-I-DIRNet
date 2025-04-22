# python imports
import os
import glob
import random
import warnings
import time
# external imports
import torch
import numpy as np
# import SimpleITK as sitk
from PIL import Image

import torch.utils.data as Data
# internal imports
from model import losses
from model.config import args
from model.datagenerators import Dataset
from model.PottsMorph_model import POTTSNET, SpatialTransformer




def save_image(img, name, slice_idx=None):
    """
    保存医学图像为 PNG 格式。如果是 3D 图像，会保存为多张 PNG 切片。
    
    :param img: 输入图像 (Tensor)，形状为 (B, C, D, H, W) 或 (B, C, H, W)。
    :param name: 保存的文件名（不含扩展名）。
    :param slice_idx: 如果是 3D 图像，可指定保存哪个切片（默认为中间切片）。
    """
    # 确保保存目录存在
    os.makedirs(args.result_dir, exist_ok=True)

    # 转换为 NumPy 格式
    img = img[0, 0, ...].cpu().detach().numpy()

    # 如果是 3D 数据，选择切片
    if img.ndim == 3:  # (D, H, W)
        if slice_idx is None:
            slice_idx = img.shape[0] // 2  # 选取中间切片
        img = img[slice_idx]  # 取出 (H, W) 切片

    # 归一化到 0-255
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    img = img.astype(np.uint8)

    # 保存 PNG
    img_pil = Image.fromarray(img)
    img_pil.save(os.path.join(args.validation_result_dir, f"{name}.png"))


def read_one_image(folder_path):
    """
    从指定的文件夹中随机读取一个PNG格式的图片，并返回该图片对象。

    参数:
        folder_path (str): 存储PNG图片的文件夹路径

    返回:
        PIL.Image: 读取的图片对象

    异常:
        ValueError: 当文件夹中没有PNG图片时抛出异常。
    """
    # 获取文件夹中所有PNG文件的完整路径列表
    image_files = [os.path.join(folder_path, f) 
                   for f in os.listdir(folder_path) 
                   if f.lower().endswith('.png')]
    
    # 检查是否至少有一个PNG图片
    if not image_files:
        raise ValueError("文件夹中没有PNG图片！")
    
    # 随机选择一个图片文件
    selected_file = random.choice(image_files)
    
    # 读取图片并返回
    image = Image.open(selected_file).convert("L")
    return image


def ssd(imgA: np.ndarray, imgB: np.ndarray) -> float:
    """
    计算 SSD(A, B) = (1/2) * sum((A_ij - B_ij)^2).
    """
    A = imgA.astype(np.float64)
    B = imgB.astype(np.float64)
    diff = A - B
    return 0.5 * np.sum(diff**2)

def compute_re_ssd(f_img: np.ndarray, m_img: np.ndarray, m2f_img: np.ndarray) -> float:
    """
    计算 Re-SSD(m, f) = SSD(m2f, f) / SSD(m, f).
    """
    denom = ssd(m_img, f_img)
    if denom == 0:
        # 如果两张图完全相同或都是常数图，则分母可能为 0
        return float('inf')
    
    numer = ssd(m2f_img, f_img)
    return numer / denom


def validation():
    # 创建需要的文件夹并指定gpu
    
   
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    

    # 读取案例图像
    folder = args.validation_dir
    img_example = read_one_image(folder)
    input_img = np.array(img_example, dtype=np.float32)  # 转换为 NumPy 数组
    input_img = input_img[np.newaxis, np.newaxis, ...]  # 添加 batch 和 channel 维度
    vol_size = input_img.shape[2:]
    # [B, C, W, H]
    

    
    UNet = POTTSNET(args).to(device)
    UNet.load_state_dict(torch.load(args.checkpoint_path))
    STN = SpatialTransformer(vol_size).to(device) # 空间变换网络, 用于将形变场施加到浮动图像上, 生成配准后的图像, vol_size = [W, H]
    UNet = UNet.float()  
    STN = STN.float()


    UNet.eval()
    STN.eval()
    
    
    # opt = Adam(UNet.parameters(), lr=args.lr)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn_1 = losses.second_order_loss
    grad_loss_fn_2 = losses.gradient_loss
    CR_loss_fn = losses.curl_regularizer
    grid_folds = losses.count_grid_folds

    # Get all the names of the validation data
    validation_files = glob.glob(os.path.join(args.validation_dir, '*.png'))
    DS = Dataset(files=validation_files)
    print("Number of validation images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Training loop.
    for i in range(1, args.n_iter + 1): # 19.22
        # Generate the moving images and convert them to tensors.

        f_img = read_one_image(folder) 
        # f_img = Image.open("D://deeplearning_for_registration//PottsMorph_3//picture_in_SIIMS2020//ball_f.png").convert("L")
        input_fixed = np.array(f_img, dtype=np.float32)  # 转换为 NumPy 数组
        input_fixed = input_fixed[np.newaxis, np.newaxis, ...]  # 添加 batch 和 channel 维度
        # vol_size = input_fixed.shape[2:]
        # [B, C, D, W, H]
        # input_fixed = np.repeat(input_fixed, args.batch_size, axis=0) # 添加 batch 维度
        input_fixed = torch.from_numpy(input_fixed).to(device).float()

        input_moving = iter(DL).next()
        # input_moving = Image.open("D://deeplearning_for_registration//PottsMorph_3//picture_in_SIIMS2020//ball_m.png").convert("L")
        # input_moving = np.array(input_moving, dtype=np.float32)  # 转换为 NumPy 数组
        # input_moving = input_moving[np.newaxis, np.newaxis, ...]

        # input_moving = torch.from_numpy(input_moving)  # 添加 batch 和 channel 维度
        
        input_moving = input_moving.to(device).float()
        input_moving_temp = input_moving.clone()

        start=time.time()

        for j in range(args.cascade_nums):
            # Run the data through the model to produce warp and flow field

            flow_m2f = UNet(input_moving, input_fixed)
            m2f = STN(input_moving, flow_m2f)
            input_moving = m2f

        # Run the data through the model to produce warp and flow field
        # flow_m2f = UNet(input_moving, input_fixed)
        # m2f = STN(input_moving, flow_m2f)

        # Calculate loss
        sim_loss = sim_loss_fn(m2f, input_fixed)
        grad_loss = grad_loss_fn_1(flow_m2f) + grad_loss_fn_2(flow_m2f) + CR_loss_fn(flow_m2f)
        loss = sim_loss + args.alpha * grad_loss
        MFN = grid_folds(flow_m2f)

        # Calculate Re-SSD
        re_ssd = compute_re_ssd(input_fixed[0, 0, ...].cpu().detach().numpy(),
                               input_moving_temp[0, 0, ...].cpu().detach().numpy(),
                               m2f[0, 0, ...].cpu().detach().numpy()) #[0, 0, ...]的作用是取出batch_size=1的图像
        # print("alpha: ", args.alpha)
        end=time.time()
        tt=end-start
        print("i: %d  loss: %f  sim: %f  grad: %f  validation_time: %f  re_ssd: %f  MFN: %d"  % (i, loss.item(), sim_loss.item(), grad_loss.item(), tt, re_ssd, MFN), flush=True)

        # Backwards and optimize
        # opt.zero_grad()
        # loss.backward()
        # opt.step()
        

        

        if i % args.n_save_iter == 0:
            # Save model checkpoint
            save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            torch.save(UNet.state_dict(), save_file_name)
            # Save images
            m_name = str(i) + "_m.png"
            m2f_name = str(i) + "_m2f.png"
            f_name = str(i) + "_f.png"
            save_image(input_fixed, f_name)
            save_image(input_moving_temp, m_name)
            save_image(m2f, m2f_name)
            print("warped images have saved.")
    
    
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    validation()