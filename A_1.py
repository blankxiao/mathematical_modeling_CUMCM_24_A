"""
@Author: blankxiao
@file: A_1.py
@Created: 2024-09-05 21:24
@Desc: A题第一问 所有点在300秒内的运动轨迹
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq, minimize, fsolve

# 模拟的时间 s
time = 300
# 螺线距离 m
spiral_distance = 0.55

# 板凳个数 第一节为龙头
num_point = 224
# 龙头长度 m
L_loong_head = 3.41
# 龙身和龙尾长度 m
L_loong_body = 2.20

# 龙头初始位置 第一圈角度为0 
theta = 0



def get_alpha_0(t: int):
    """
    获取龙头的角度
    @param t: 时间s
    @return: alpha0 角度 
    """
    # spiral_distance 为螺距
    return 32 * np.pi - np.sqrt((32 * np.pi) ** 2 - 4 * np.pi / spiral_distance * t)

def get_r_0(alpha_0: int):
    """
    @param alpha_0: alpha0角度
    @return: r0 极径
    """
    return 16 * spiral_distance - spiral_distance / (2 * np.pi) * alpha_0


def get_alpha_i(r_i: int):
    """
    获取r_i的半径
    @param alpha_i: 角度
    @return: r_i 半径 m
    """
    return 32 * np.pi - 2 * np.pi / spiral_distance * r_i

def get_r_i(r_i_pre: int, point_index: int):
    """
    获取alpha_i的半径
    @param r_i: 半径 cm
    @return: alpha_i 角度
    """
    bandeng_len = L_loong_body
    if point_index == 1:
        bandeng_len = L_loong_head

    def equation(r_i):
        return r_i_pre ** 2 + r_i ** 2 - 2 * r_i_pre * r_i * np.cos(2 * np.pi / spiral_distance * (r_i - r_i_pre)) - bandeng_len ** 2

    r_i_solution = fsolve(equation, r_i_pre)

    return r_i_solution[0]


def plot_polar_scatter(theta, r, color='blue', marker='o', s=30, alpha=0.8, figsize=(8, 6), line_color='red', line_width=1):
    """
    绘制极坐标散点图，并连接每个点

    theta: 角度值的列表（弧度制）
    r: 半径值的列表
    color: 散点颜色
    marker: 散点形状
    s: 散点大小
    alpha: 散点透明度
    figsize: 图像大小
    line_color: 连接线的颜色
    line_width: 连接线的宽度
    """
    plt.figure(figsize=figsize)
    ax = plt.subplot(111, projection='polar')  # 创建极坐标子图
    
    # 绘制散点
    ax.scatter(theta, r, color=color, marker=marker, s=s, alpha=alpha)
    
    # 绘制连接线
    ax.plot(theta, r, color=line_color, linewidth=line_width)
    
    plt.grid(True)
    plt.show()

def get_x_y(alpha, r):
    """
    获取x,y坐标
    @param alpha: 角度
    @param r: 半径
    @return: x,y坐标
    """
    return r * sp.cos(-alpha), r * sp.sin(-alpha)


df_xy = pd.DataFrame(index=pd.MultiIndex.from_product([range(num_point), ['x', 'y']]), columns=range(time + 1))
df_ra = pd.DataFrame(index=pd.MultiIndex.from_product([range(num_point), ['r', 'alpha']]), columns=range(time + 1))

for point_index in range(num_point + 1):
    for t in range(10):
        if point_index != 0:
            cur_r = get_r_i(r_i_pre=df_ra[t][point_index - 1, 'r'], point_index=point_index)
            cur_alpha = get_alpha_i(r_i=cur_r) 
        else:
            cur_alpha = get_alpha_0(t)
            cur_r = get_r_0(cur_alpha)

        cur_x, cur_y = get_x_y(alpha=cur_alpha, r=cur_r)
        
        # 将 x 和 y 值分别添加到对应的 t 列中
        df_xy.at[(point_index, 'x'), t] = cur_x
        df_xy.at[(point_index, 'y'), t] = cur_y
        
        # 记录当前的 r 和 alpha
        df_ra.at[(point_index, 'r'), t] = cur_r
        df_ra.at[(point_index, 'alpha'), t] = cur_alpha

df_xy.to_csv('df_xy.csv')
df_ra.to_csv('df_ra.csv')

# plot_polar_scatter(alpha_list, r_list)



