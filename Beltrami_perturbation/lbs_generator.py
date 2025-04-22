import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from imageio import imwrite
from beltrami_generation_function import beltrami_generation_function
from lbs_function import lbs_function
from Recombination import recombination
# 该脚本用于读取两个文件夹中的.mat文件，对每对文件生成多重扰动形变并保存结果。

# 导入先前实现的函数
# from beltrami_py import beltrami_generation_function
# from lbs_function_py import lbs_function
# from recombination_py import recombination


def lbs_generator(folderPath1: str,
                  folderPath2: str,
                  savepath: str,
                  start_idx: int = 2,
                  end_idx: int = 50):
    """
    读取两个文件夹中的.mat文件，对每对文件生成多重扰动形变并保存结果。

    Args:
        folderPath1: 包含原图像.mat文件的文件夹路径
        folderPath2: 包含形变场.mat文件的文件夹路径
        savepath   : 结果保存目录
        start_idx  : 开始索引 (0-based, 默认2 对应 MATLAB kkk=3)
        end_idx    : 结束索引 (exclusive, 默认50 对应 MATLAB kkk<51)
    """
    os.makedirs(savepath, exist_ok=True)

    # 获取所有.mat文件并排序
    files1 = sorted(glob.glob(os.path.join(folderPath1, '*.mat')))
    files2 = sorted(glob.glob(os.path.join(folderPath2, '*.mat')))

    N1 = N2 = 129

    # 遍历文件对
    for kkk in range(start_idx, min(end_idx, len(files1))):
        mat1 = sio.loadmat(files1[kkk])
        mat2 = sio.loadmat(files2[kkk])
        # 假设.mat文件中变量名为 'k' 和 'phyx','phyy'
        T = mat1.get('k')
        phyx = mat2.get('phyx')
        phyy = mat2.get('phyy')

        # 计算初始Beltrami系数
        D2u, D1u = np.gradient(phyx)
        D2v, D1v = np.gradient(phyy)
        rou0 = (D1u**2 - D2u**2 + D1v**2 - D2v**2) / ((D1u + D2v)**2 + (D2u - D1v)**2)
        tau0 = 2*(D1u*D2u + D1v*D2v) / ((D1u + D2v)**2 + (D2u - D1v)**2)
        if np.max(rou0**2 + tau0**2) >= 1:
            break

        base_name = os.path.splitext(os.path.basename(files1[kkk]))[0]
        # 保存原图像为 JPG
        imwrite(os.path.join(savepath, f"{base_name}.jpg"), T)

        # 多次扰动与求解
        for m in range(1, 1001):
            rou, tau = beltrami_generation_function(rou0, tau0, N1, N2)
            phyx_new, phyy_new, ssd_mu = lbs_function(rou, tau, 0.0003)
            if ssd_mu < 30:
                # 计算Jacobian行列式范围
                D2u2, D1u2 = np.gradient(phyx_new)
                D2v2, D1v2 = np.gradient(phyy_new)
                J = -D2u2 * D1v2 + D1u2 * D2v2
                print(f"det|J| range: {J.max():.6f}, {J.min():.6f}")

                # 复合形变
                D = recombination(T, phyx_new, phyy_new)

                # 绘图但不显示
                plt.figure(figsize=(6,6))
                plt.imshow(D, cmap='gray', origin='lower')
                plt.axis('equal')
                plt.xlim(1, N2)
                plt.ylim(1, N1)
                # 绘制网格线
                for i in range(0, N1, 3):
                    plt.plot(phyy_new[i, :], phyx_new[i, :], 'b')
                for j in range(0, N2, 3):
                    plt.plot(phyy_new[:, j], phyx_new[:, j], 'b')

                # 保存结果
                out_mat = {
                    'T': T,
                    'D': D,
                    'phyx': phyx_new,
                    'phyy': phyy_new,
                    'rou': rou,
                    'tau': tau
                }
                mat_name = f"{base_name}_{m}_{ssd_mu:.4f}.mat"
                sio.savemat(os.path.join(savepath, mat_name), out_mat)
                jpg_name = f"{base_name}_{m}_{ssd_mu:.4f}_{J.min():.4f}.jpg"
                plt.savefig(os.path.join(savepath, jpg_name))
                plt.close()


if __name__ == '__main__':
    folderPath1 = r'E://研究生代码项目//field_generation//mat360//mat360'
    folderPath2 = r'E://研究生代码项目//field_generation//k=1000//k=1000'
    savepath = r'E://研究生代码项目//field_generation//lbs//lbs//随机多峰构造//50'
    lbs_generator(folderPath1, folderPath2, savepath)
