# C-I-DIRNet

This is the official Pytorch implementation of "An interpretable neural network for diffeomorphic image registration: theory and application" by Huan Han  et al. (Accepted to Inverse Problems) 

## Framework



<img src="C:\Users\hr_ha\AppData\Roaming\Typora\typora-user-images\image-20260510231503343.png" alt="image-20260510231503343" style="zoom:67%;" />



## Environment

The code was tested under the following environment:

| Package | Version |
| ------- | ------- |
| Python  | 3.13.5  |
| PyTorch | 2.8.0   |
| CUDA    | 12.9    |

## Project Structure

```text
C-I-DIRNet/
├── train.py          # Training script
├── test.py           # Testing script
├── generator.py      # Generate training data for the next-level network
├── model/            # Model-related files
├── data/             # Dataset folder
├── Checkpoint/       # Saved checkpoints
├── Result/           # Experimental results
└── README.md
```

## Data

