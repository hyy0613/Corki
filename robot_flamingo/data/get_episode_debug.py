import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.interpolate import interp1d
import torch

file_dir = Path("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/data/robotics/task_D_D/training/")
annotation_file = os.path.join(file_dir,'lang_annotations/auto_lang_ann.npy')
lang_informations = np.load(annotation_file,allow_pickle=True).item()
indx_episodes = lang_informations["info"]["indx"]

data_dir ="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/data/robotics/task_D_D/training"
#
# for (begin,end) in indx_episodes:
begin,end = indx_episodes[0]
information = []
for i in range(begin,begin+5):
    episode_name = f"episode_{i:07d}.npz"
    information.append(np.load(os.path.join(data_dir,episode_name),allow_pickle=True)['rel_actions'][0])

print(information)
#%%

# b = torch.tensor(information, dtype=torch.float32).reshape(5,1)
# t = torch.matmul(A,b).numpy().squeeze(1)

# # 原始数据点
# x = np.array([-2,-1,0,1,2])
# x_repeated = np.tile(x, (36, 1)).reshape(6,6,5)
# print(x_repeated)
#
# bs = x.shape[0]
# num = x.shape[1]
#
# x_flat = x.view(bs * num, -1)
# y_flat = y.view(bs * num, -1)
# y = np.array(t)
#%%
import numpy as np
from scipy.optimize import curve_fit
x = np.array(range(5))
y = information
# y = np.cumsum(information)

# def cubic_function(x, a, b, c, d):
#     return a * x**3 + b * x**2 + c * x + d
#
# # 使用curve_fit进行拟合
# popt, pcov = curve_fit(cubic_function, x, y)
#
# # 生成拟合曲线上的点
# x_fit = np.linspace(x[0], x[-1], 100)
# y_fit = cubic_function(x_fit, *popt)
#
# # 绘制原始离散点和拟合曲线
# plt.scatter(x, y, label='Original Points')
# plt.plot(x_fit, y_fit, color='r', label='Fitted Curve')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

coefficients = np.polyfit(x,y, 3)
print(coefficients)
poly_model = np.poly1d(coefficients)
#
#
#
# # 生成平滑曲线上的更多点
x_smooth = np.linspace(x.min(), x.max(), 100)
y_smooth = poly_model(x_smooth)
#
# # 绘制原始数据点和平滑曲线
plt.scatter(x, y, label='Original Data')
plt.plot(x_smooth, y_smooth, color='red', label='Smoot Curve')
#
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
#%%
params[i, :] = coefficients
x_repeated = np.tile(x, (36, 1)).reshape(6,6,5)
x_tensor = torch.tensor(x_repeated).cuda().float()

bs = x_repeated.shape[0]
num = x_repeated.shape[1]

def cubic_function(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# 定义损失函数，即观测值与拟合值之间的残差平方和
def loss_function(params, x, y):
    a, b, c, d = params
    return cubic_function(x, a, b, c, d) - y

for i in range(0, c.size(1) - 2):
    k = max(i-2,0)
    inputs = c[:, k:i+3, :]
    width = max(-3-i,-5)
    inputs_trans= inputs.transpose(1,2)
    targets = torch.matmul(inputs_trans,A.T[width:,width:]).cuda().float()
    points = targets.shape[-1]
    x_flat = x_tensor.view(bs * num, -1)[-points:]
    y_flat = targets.view(bs * num,-1)

    params_guess = torch.tensor(np.tile(np.array([1, 1, 1, 1]), (36, 1)).reshape(36, 4)).cuda().float()  # 初始参数猜测值

    # 使用最小二乘法进行拟合
    params_fit, _ = torch.nn.functional.least_squares(loss_function, params_guess, args=(x_flat, y_flat))
    print(params_fit)



# # 初始化拟合参数
# params_guess = torch.tensor(np.tile(np.array([1,1,1,1]),(36, 1)).reshape(36,4)) # 初始参数猜测值
#
# # 使用最小二乘法进行拟合
# params_fit, _ = torch.nn.functional.least_squares(loss_function, params_guess, args=(x, y))

# 打印所有曲线的拟合参数
# print("Parameters for all curves:")
# print(params_fit)
#
#
# # 多项式拟合
# degree = 4  # 多项式的阶数
# coefficients = np.polyfit(x, y, degree)  # 使用最小二乘法进行多项式拟合
# print(coefficients)
#
# poly_model = np.poly1d(coefficients)
#
#
# f = interp1d(x, y, kind='cubic')  # 使用三次样条插值
#
# # 生成平滑曲线上的更多点
# x_smooth = np.linspace(x.min(), x.max(), 100)
# y_smooth = poly_model(x_smooth)
#
# # 绘制原始数据点和平滑曲线
# plt.scatter(x, y, label='Original Data')
# plt.plot(x_smooth, y_smooth, color='red', label='Smoot Curve')
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()
#%%

import torch

# 假设你有一个张量
tensor = torch.randn(5,6)
print(tensor)

# 使用torch.diff函数对每个元素减去前一个元素
diff_tensor = torch.diff(tensor)

# 打印计算结果
print(diff_tensor)
#%%
import torch

# 假设你有一个大小为5x6的张量
tensor = torch.tensor([[1, 2, 3, 4, 5, 6],
                       [2, 4, 6, 8, 10, 12],
                       [3, 6, 9, 12, 15, 18],
                       [4, 8, 12, 16, 20, 24],
                       [5, 10, 15, 20, 25, 30]])

# 使用torch.diff函数对每一行的元素减去前一个元素
diff_tensor = torch.cat((tensor[:, :1], torch.diff(tensor, dim=1)), dim=1)

# 打印计算结果
print(diff_tensor)
#%%
import torch

# 假设你有一个形状为(6, 12, 5, 6)的张量
tensor = torch.randn(1,1,5, 6)

print(tensor)
# 使用torch.cumsum函数对最后一维进行累加求和
cumsum_tensor = torch.cumsum(tensor, dim=-2)

# 打印累加求和后的张量
print(cumsum_tensor)
#%%
import torch
import numpy as np

tensor = torch.randn((2,5,6))
import torch

# 原始张量
# x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
x = torch.randn((2,5,6))

# 交换最后两个维度
x_transposed = torch.transpose(x, dim0=-1, dim1=-2).cuda()
bs,ac = x_transposed.shape[0],x_transposed.shape[1]
x_transposed = x_transposed.reshape(bs*ac,-1)
print(x_transposed.shape)
print(x_transposed[0].cpu().numpy().shape)
k = np.array([-2,-1,0,1,2])
print(k.shape)
coefficients = np.polyfit(k ,x_transposed[0].cpu().numpy(), 3)
print(coefficients)
#%%
a = torch.randn(6,1,4)
b = torch.randn(6,1,4)
c = torch.stack([a,b],dim=-2)
print(a)
print(b)
print(c)
print(c.shape)
#%%
import torch
input_feature = torch.randn(6,12,1)
random_matrix = torch.tensor([1, 0], device=input_feature.device).repeat(input_feature.shape[0], input_feature.shape[1] // 2 + 1)[:input_feature.shape[0], :input_feature.shape[1]]
print(random_matrix)
#%%
import torch

# 假设有一个三维张量 tensor，形状为 (batch_size, height, width)
tensor = torch.tensor([[[1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10]],
                       [[11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20]]])

# 在第二维进行隔着采样，步长为2
print(tensor.shape)
sampled_tensor = torch.index_select(tensor, dim=1, index=torch.arange(0, tensor.size(1), 2))

print(sampled_tensor.shape)
#%%
print(12//2)


