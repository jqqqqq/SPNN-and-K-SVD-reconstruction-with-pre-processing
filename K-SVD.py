import numpy as np
from sklearn.linear_model import orthogonal_mp
from skimage import io, color, util
import matplotlib.pyplot as plt
import csv
import os
import torch
import time
from datetime import datetime
import csv
import pandas as pd

def save_dictionary_to_csv(dictionary, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"Atom {i}" for i in range(dictionary.shape[1])])  # 添加列名
        for row in dictionary:
            writer.writerow(row)
def save_sparsecode_to_csv(sparsecode, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"Component {i}" for i in range(sparsecode.shape[1])])  # 添加列名
        for row in sparsecode:
            writer.writerow(row)

def rand(max):
    random_numbers = (max * (1 - 2 * torch.rand(1)))
    return random_numbers


def save_accuracy_to_csv(iteration, error):
    t = datetime.fromtimestamp(time.time())
    filename = f"iteration_error_{datetime.strftime(t, '%m%d%H')}.csv"
    # 提取坐标和特征
    iteration = iteration + 1  # 提取坐标
    error = error  # 提取非零值
    # 保存到 CSV 文件
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "error"])  # 根据实际维度调整列名
        for i in range(iteration.shape[0]):
            writer.writerow([*iteration[i], *error[i]])  # 将坐标和对应特征值保存

class KSVD(object):
    def __init__(self, n_components, max_iter=1e+6, tol=1e-6, n_nonzero_coefs=None):
        """
        稀疏模型 Y = DX，Y为样本矩阵，使用K-SVD动态更新字典矩阵D和稀疏矩阵X
        :param n_components: 字典所含原子个数（字典的列数）
        :param max_iter: 最大迭代次数
        :param tol: 稀疏表示结果的容差
        :param n_nonzero_coefs: 稀疏度
        """
        self.dictionary = None
        self.sparsecode = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs

    def _initialize(self, y):
        """
        初始化字典矩阵，采用SVD分解
        """
        u, s, v = np.linalg.svd(y, full_matrices=False)
        self.dictionary = u[:, :self.n_components]

    def _update_dict(self, y, d, x):
        """
        使用K-SVD更新字典的过程
        """
        for i in range(self.n_components):
            idx = np.nonzero(x[i, :])[0]
            if len(idx) == 0:
                continue

            d[:, i] = 0
            r = (y - np.dot(d, x))[:, idx] #残差r
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0]
            x[i, idx] = s[0] * v[0, :]
        return d, x

    def fit(self, y):
        """
        K-SVD 迭代过程
        """
        self._initialize(y)
        for iteration in range(200):
            x = orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
            error = (np.linalg.norm(y - np.dot(self.dictionary, x))) * 0.001
            print(f"Iteration {iteration + 1}, reconstruction error: {error}")
            # save_accuracy_to_csv(iteration, error)
            #if error < self.tol:
                #break
            self.dictionary, x = self._update_dict(y, self.dictionary, x)

        self.sparsecode = orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
        return self.dictionary, self.sparsecode

def extract_blocks(img, block_size):
    """
    将图像分块为不重叠的小块
    """
    h, w = img.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i + block_size, j:j + block_size]
            blocks.append(block.flatten())
    return np.array(blocks)

def reconstruct_image(blocks, img_shape, block_size):
    """
    从块还原图像
    """
    h, w = img_shape
    reconstructed = np.zeros(img_shape)
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            reconstructed[i:i + block_size, j:j + block_size] = blocks[idx].reshape((block_size, block_size))
            idx += 1
    return reconstructed


def save_blocks(blocks, block_size, output_dir, blocks_per_file=64):
    """
    将每64个分块保存为一个本地图像文件
    :param blocks: 分块后的图像数组
    :param block_size: 每个分块的尺寸
    :param output_dir: 输出目录
    :param blocks_per_file: 每个文件保存的分块数量
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 计算需要保存的文件数量
    num_files = int(np.ceil(len(blocks) / blocks_per_file))

    for file_idx in range(num_files):
        # 计算当前文件的起始和结束索引
        start_idx = file_idx * blocks_per_file
        end_idx = min((file_idx + 1) * blocks_per_file, len(blocks))

        # 计算画布的行数和列数
        num_blocks_in_file = end_idx - start_idx
        cols = min(8, num_blocks_in_file)  # 每行最多8个块
        rows = int(np.ceil(num_blocks_in_file / cols))  # 行数

        # 创建一个画布，用于存放当前文件的所有块
        canvas_height = block_size * rows
        canvas_width = block_size * cols
        canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

        # 将块填充到画布上
        for i in range(num_blocks_in_file):
            block_idx = start_idx + i
            block = blocks[block_idx].reshape((block_size, block_size))
            row = i // cols
            col = i % cols
            canvas[row * block_size:(row + 1) * block_size, col * block_size:(col + 1) * block_size] = block

        # 构建文件路径
        file_path = os.path.join(output_dir, f"blocks_{file_idx + 1}.png")

        # 保存画布为图像文件
        plt.imsave(file_path, canvas, cmap='gray')



def display_blocks(blocks, block_size, blocks_per_row=8):
    """
    显示分块图像
    """
    num_blocks = len(blocks)
    rows = int(np.ceil(num_blocks / blocks_per_row))
    fig, axes = plt.subplots(rows, blocks_per_row, figsize=(12, 6))
    for idx, block in enumerate(blocks):
        row, col = divmod(idx, blocks_per_row)
        axes[row, col].imshow(block, cmap='gray')
        axes[row, col].axis('off')
    for idx in range(num_blocks, rows * blocks_per_row):
        row, col = divmod(idx, blocks_per_row)
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()


# 加载本地图像
image_path = "xiangchang.jpg"  # 替换为实际图像路径
image = io.imread(image_path)

# 转换为灰度图像
if image.ndim == 3:
    image = color.rgb2gray(image)

#t = 1e-6
t = 2
x = image #归一化
'''
n=50仿真标准差
'''
data = pd.read_csv('file.csv')

data_min = data.min()
data_max = data.max()
data_range = data_max - data_min

# 使用条件判断防止分母为零
min_val = 1e-6  # 阈值
if data_range < min_val:
    data_range = min_val

data = 1 - ((data - data_min) / data_range)


# 归一化到 [0, 1]
image = data * 255

# 确保图像大小是 block_size 的整数倍
block_size = 8
image = util.crop(image, ((0, image.shape[0] % block_size), (0, image.shape[1] % block_size)))

# 提取图像块
blocks = extract_blocks(image, block_size)

# 检查 n_components 是否超界
n_components = min(100, block_size ** 2)  # 保证字典原子数小于等于块维度
ksvd = KSVD(n_components=n_components, max_iter=10, n_nonzero_coefs=4)

# 使用 KSVD 进行字典学习
dictionary, sparsecode = ksvd.fit(blocks.T)

# 使用字典和稀疏表示重建图像块
reconstructed_blocks = dictionary.dot(sparsecode).T

# 确保稀疏编码索引安全
for i in range(ksvd.n_components):
    idx = np.nonzero(sparsecode[i, :])[0]
    if len(idx) == 0:
        print(f"Skipping dictionary atom {i}, no corresponding sparse coefficients.")

t = datetime.fromtimestamp(time.time())

# 重建图像
reconstructed_image = reconstruct_image(reconstructed_blocks, image.shape, block_size)


plt.imshow(image, cmap='gray')
plt.axis('off')
plt.savefig(f"image_{datetime.strftime(t, '%m%d%H')}.png")
plt.show()

#plt.title("Reconstructed Image")
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')
plt.savefig(f"Reconstructed_{datetime.strftime(t, '%m%d%H')}.png")
plt.show()

#plt.title("Learned Dictionary Atoms")
plt.imshow(dictionary, cmap='gray', aspect='auto')
plt.axis('off')
plt.savefig(f"Dictionary_{datetime.strftime(t, '%m%d%H')}.png")
plt.show()

save_dictionary_to_csv(dictionary, "dictionary.csv")
save_sparsecode_to_csv(sparsecode, "sparsecode.csv")

output_dir = "output_blocks"  # 保存路径
save_blocks(blocks, block_size, output_dir)

# 显示分块
display_blocks(blocks, block_size)
