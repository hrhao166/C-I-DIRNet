import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import os
import random
from PIL import Image
import numpy as np

from model.PottsMorph_model import POTTSNET, SpatialTransformer, read_one_image
from model.config import args
from model.PottsMorph_model import POTTS, layer1
from model.datagenerators import Dataset

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


# 将形变场应用到源图像上，得到配准后的图像
def apply_deformation(src, flow):
    """
    Apply the deformation field to the source image.

    :param src: Source image tensor of shape (B, C, H, W) or (B, C, D, H, W)
    :param flow: Deformation field tensor of shape (B, 2, H, W) or (B, 3, D, H, W)
    :return: Deformed image tensor of the same shape as src
    """
    spatial_transformer = SpatialTransformer(src.shape[2:], mode='bilinear')
    deformed_image = spatial_transformer(src, flow)

    return deformed_image


# 将配准后的图像作为浮动图像，固定图像不变，再次输入POTTSNET进行配准，取名为cascaded_pottsnet
class cascaded_pottsnet(POTTS):
# assemble blocks
    def __init__(self, args=None, in_channels=1):
        super(cascaded_pottsnet,self).__init__(args=args)
        
        
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        
        
        
        UNet = POTTSNET(args).to(self.device)
        UNet.load_state_dict(torch.load(args.checkpoint_path))
        STN = SpatialTransformer(vol_size).to(self.device)

    def forward(self, src, tgt):

        x = torch.cat([src, tgt], dim=1) # 拼接浮动图像和目标图像, dim=1表示在通道维度上拼接
        
        out=self.layer1(x) # x是原始图像对, layer1是对其进行预处理，out是形变场的初始值
        for idx in range(self.cascade_nums):
            out=self.blocks[idx](out,src,tgt) # out是形变场的输出

        return out # out是形变场的最终值