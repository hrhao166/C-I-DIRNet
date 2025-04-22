import argparse
import torch

parser = argparse.ArgumentParser()

# 公共参数
parser.add_argument("--device", type=str, help="device to use",
                    dest="device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='0')
parser.add_argument("--atlas_file", type=str, help="gpu id number",
                    dest="atlas_file", default='D://deeplearning_for_registration//PottsMorph_3//data//fixed//extracted_image_1.png')
parser.add_argument("--model", type=str, help="voxelmorph 1 or 2",
                    dest="model", choices=['vm1', 'vm2'], default='vm2')
parser.add_argument("--result_dir", type=str, help="results folder",
                    dest="result_dir", default='D://deeplearning_for_registration//PottsMorph_3//Result')
parser.add_argument("--validation_result_dir", type=str, help="validation folder",
                    dest="validation_result_dir", default='D://deeplearning_for_registration//PottsMorph_3//validation_result')
parser.add_argument("--mid_channels", type=list, help="number of channels in the middle layers",
                    dest="mid_channels", default=[16, 32, 64]) # mid_channels的值是list类型，里面是int类型的数字
parser.add_argument("--times_list", type=list, help="number of times for each layer",
                    dest="times_list", default=[1, 1, 1]) # times_list的值是list类型，里面是int类型的数字
parser.add_argument("--tau", type=float, help="times step size",
                    dest="tau", default=0.5)
parser.add_argument("--Theta", type=float, help="CR constraint term",
                    dest="Theta", default=0.1)
parser.add_argument("--num_blocks", type=int, help="number of blocks",
                    dest="num_blocks", default=2)
parser.add_argument("--cascade_nums", type=int, help="number of cascades",
                    dest="cascade_nums", default=10)
parser.add_argument("--kernel_size_bound", type=int, help="largest kernel size allowed",
                    dest="kernel_size_bound", default=3)
parser.add_argument("--dim", type=int, help="dimension of the image",
                    dest="dim", default=2)
parser.add_argument("--connect", type=bool, help="True if use skip-connections between encoder and decoder",
                    dest="connect", default=True) #  connect的值是bool类型，True和False
parser.add_argument("--tau_explicit", type=bool, help="True if use explicit tau",
                    dest="tau_explicit", default=True)
parser.add_argument("--BNLearn", type=bool, help="True if learn parameters in batch normalization",
                    dest="BNLearn", default=True)
parser.add_argument("--bn", type=bool, help="True if use batch normalization",
                    dest="bn", default=True)

# train时参数
parser.add_argument("--train_dir", type=str, help="data folder with training vols",
                    dest="train_dir", default="D://deeplearning_for_registration//PottsMorph_3//data//train_small")
parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=4e-4)
parser.add_argument("--n_iter", type=int, help="number of iterations",
                    dest="n_iter", default=3500)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc",
                    dest="sim_loss", default='ncc')
parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=0.1)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
parser.add_argument("--n_save_iter", type=int, help="frequency of model saves",
                    dest="n_save_iter", default=100)
parser.add_argument("--model_dir", type=str, help="models folder",
                    dest="model_dir", default='D://deeplearning_for_registration//PottsMorph_3//Checkpoint')
parser.add_argument("--log_dir", type=str, help="logs folder",
                    dest="log_dir", default='D://deeplearning_for_registration//PottsMorph_3//Log')

# validation时参数
parser.add_argument("--validation_dir", type=str, help="validation data directory",
                    dest="validation_dir", default='D://deeplearning_for_registration//PottsMorph_3//data//validation')


# test时参数
parser.add_argument("--test_dir", type=str, help="test data directory",
                    dest="test_dir", default='D://deeplearning_for_registration//VoxelMorph-torch-master//VoxelMorph-torch-master//data//test')
parser.add_argument("--label_dir", type=str, help="label data directory",
                    dest="label_dir", default='D://deeplearning_for_registration//VoxelMorph-torch-master//VoxelMorph-torch-master//data//label')
parser.add_argument("--checkpoint_path", type=str, help="model weight file",
                    dest="checkpoint_path", default="D://deeplearning_for_registration//PottsMorph_3//Checkpoint//1500.pth")

args = parser.parse_args()