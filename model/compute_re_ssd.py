import os
import re
import cv2
import numpy as np

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

def compute_re_ssd_for_folder(folder: str):
    """
    遍历给定文件夹，查找形如: <prefix>_f.png[.png], <prefix>_m.png[.png],
    <prefix>_m2f.png[.png] 的文件（支持双重扩展名），
    三张图像一组，计算并输出对应的 Re-SSD，
    最后计算并打印所有组的平均 Re-SSD 值。
    """
    # 修改正则表达式，支持单重或双重扩展名，如 "30_f.png" 或 "30_f.png.png"
    pattern = re.compile(r'^(\d+)_(f|m|m2f)\.png(\.png)?$', re.IGNORECASE)
    
    # 用于收集同一 prefix 下的各类型文件
    data = {}

    file_list = os.listdir(folder)
    # print("文件夹中的所有文件:", file_list)  # 调试信息

    for fname in file_list:
        match = pattern.match(fname)
        if not match:
            continue
        prefix = match.group(1)
        ftype  = match.group(2).lower()  # 统一为小写
        # 打印匹配信息调试
        # print(f"匹配到文件: {fname} => prefix: {prefix}, type: {ftype}")
        if prefix not in data:
            data[prefix] = {}
        data[prefix][ftype] = fname

    # print("收集到的图像分组:", data)  # 调试输出

    # 用于保存所有计算得到的 Re-SSD 值
    re_ssd_values = []

    # 依次遍历各组
    for prefix, group in data.items():
        if all(key in group for key in ['f', 'm', 'm2f']):
            f_path   = os.path.join(folder, group['f'])
            m_path   = os.path.join(folder, group['m'])
            m2f_path = os.path.join(folder, group['m2f'])

            # 读取灰度图
            f_img   = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
            m_img   = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            m2f_img = cv2.imread(m2f_path, cv2.IMREAD_GRAYSCALE)

            if f_img is None or m_img is None or m2f_img is None:
                print(f"警告: 图像读取失败, prefix={prefix}")
                continue
            if not (f_img.shape == m_img.shape == m2f_img.shape):
                print(f"警告: 图像尺寸不一致, prefix={prefix}")
                continue

            val = compute_re_ssd(f_img, m_img, m2f_img)
            print(f"prefix={prefix}, Re-SSD = {val:.6f}")
            re_ssd_values.append(val)

    # 计算并打印所有组的平均 Re-SSD
    if re_ssd_values:
        average = np.mean(re_ssd_values)
        print(f"\n所有组的平均 Re-SSD = {average:.6f}")
    else:
        print("没有找到符合条件的图像组合进行计算。")

def main():
    # 设置文件夹路径，根据实际情况调整
    folder_path = r"D:\deeplearning_for_registration\PottsMorph_3\validation_result"
    
    # 计算并输出结果
    compute_re_ssd_for_folder(folder_path)

if __name__ == '__main__':
    main()
