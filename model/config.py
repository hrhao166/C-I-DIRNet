import argparse
import torch

parser = argparse.ArgumentParser()

# Common parameters
parser.add_argument("--device", type=str, help="device to use",
                    dest="device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='0')
parser.add_argument("--result_dir", type=str, help="results folder",
                    dest="result_dir", default='./Result')
parser.add_argument("--mid_channels", type=list, help="number of channels in the middle layers",
                    dest="mid_channels", default=[16, 32, 64]) # The value of mid_channels is a list containing int values
parser.add_argument("--times_list", type=list, help="number of times for each layer",
                    dest="times_list", default=[1, 1, 1]) # The value of times_list is a list containing int values
parser.add_argument("--tau", type=float, help="times step size",
                    dest="tau", default=0.1)
parser.add_argument("--Theta", type=float, help="CR constraint term",
                    dest="Theta", default=250)
parser.add_argument("--num_blocks", type=int, help="number of blocks",
                    dest="num_blocks", default=2)
parser.add_argument("--cascade_nums", type=int, help="number of cascades",
                    dest="cascade_nums", default=10)
parser.add_argument("--kernel_size_bound", type=int, help="largest kernel size allowed",
                    dest="kernel_size_bound", default=3)
parser.add_argument("--n_iter", type=int, help="number of iterations",
                    dest="n_iter", default=1500)
parser.add_argument("--dim", type=int, help="dimension of the image",
                    dest="dim", default=2)
parser.add_argument("--connect", type=bool, help="True if use skip-connections between encoder and decoder",
                    dest="connect", default=True) 
parser.add_argument("--tau_explicit", type=bool, help="True if use explicit tau",
                    dest="tau_explicit", default=True)
parser.add_argument("--BNLearn", type=bool, help="True if learn parameters in batch normalization",
                    dest="BNLearn", default=True)
parser.add_argument("--bn", type=bool, help="True if use batch normalization",
                    dest="bn", default=True)

# Parameters for train
parser.add_argument("--supervised_train_dir", type=str, help="data folder with training vols",
                    dest="supervised_train_dir", default=r"D:\PottsMorph_data\bg_150.h5")
parser.add_argument("--unsupervised_train_dir", type=str, help="data folder with training vols",
                    dest="unsupervised_train_dir", default=r"D:\PottsMorph_data\bg_150.h5") 
# The data is too large, so the h5 files are all placed in the supervised-learning folder
parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=4e-4)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: ncc or mse",
                    dest="sim_loss", default='ncc')
parser.add_argument("--lambda_1", type=float, help="figure similarity loss weight",
                    dest="lambda_1", default=10)  
parser.add_argument("--lambda_2", type=float, help="field similarity loss weight",
                    dest="lambda_2", default=1)  
parser.add_argument("--alpha_1", type=float, help="gradient regularization parameter",
                    dest="alpha_1", default=0)
  
parser.add_argument("--supervised_alpha_2", type=list, help="CR regularization parameter in supervised training",
                    dest="supervised_alpha_2", default=[15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])  
parser.add_argument("--unsupervised_alpha_2", type=list, help="CR regularization parameter in unsupervised training",
                    dest="unsupervised_alpha_2", default=[15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])  

parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
parser.add_argument("--n_save_iter_in_train", type=int, help="frequency of model saves",
                    dest="n_save_iter_in_train", default=1000)
parser.add_argument("--supervised_model_dir", type=str, help="models folder",
                    dest="supervised_model_dir", default='./Checkpoint/no_regularization') 
parser.add_argument("--unsupervised_model_dir", type=str, help="models folder", 
                    dest="unsupervised_model_dir", default='./Checkpoint/unsupervised_64')
parser.add_argument("--train_mode", type=bool, help="supervised training mode (True) or unsupervised training mode (False)",
                    dest="train_mode", default=True)


# Parameters for generation
parser.add_argument("--n_save_iter_in_generation", type=int, help="frequency of model saves",
                    dest="n_save_iter_in_generation", default=1)
parser.add_argument("--supervised_generation_dir", type=str, help="generation data directory",
                    dest="supervised_generation_dir", default='./data/generate_data/supervised')
parser.add_argument("--unsupervised_generation_dir", type=str, help="generation data directory",
                    dest="unsupervised_generation_dir", default='./data/generate_data/unsupervised')


# Parameters for test
parser.add_argument("--test_dir", type=str, help="test data directory",
                    dest="test_dir", default='./data/test_data')

parser.add_argument("--n_iter_in_test", type=int, help="number of iterations in test",
                    dest="n_iter_in_test", default=50)


# Model weight file paths, used during generation and test
supervised_base_path = "./Checkpoint/no_regularization/"

# Add checkpoint parameters for supervised networks at levels 1-10
for level in range(1, 11):
    parser.add_argument(
        f"--supervised_checkpoint{level}_path", 
        type=str, 
        help=f"Uet{level} weight file",
        dest=f"supervised_checkpoint{level}_path", 
        default=f"{supervised_base_path}150_no_regularization_supervised_cascade{level}_1500.pth"
    )

unsupervised_base_path = "./Checkpoint/unsupervised_64/"

# Add checkpoint parameters for unsupervised networks at levels 1-9
for level in range(1, 11):
    parser.add_argument(
        f"--unsupervised_checkpoint{level}_path", 
        type=str, 
        help=f"Uet{level} weight file",
        dest=f"unsupervised_checkpoint{level}_path", 
        default=f"{unsupervised_base_path}80_64_unsupervised_cascade{level}_1500.pth"
    )


args = parser.parse_args()