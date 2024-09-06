"""
@Author: blankxiao
@file: A_2.py
@Created: 2024-09-06 14:03
@Desc: A_2 求出碰撞时间 和当时各个部分的状态(位置和速度)
主要是求碰撞时间 然后代入A_1模型 得出碰撞后各个部分的位置和速度
"""
import sys

import pandas as pd
import numpy as np
import sympy as sp

from A_1 import get_alpha_0, get_r_0, get_x_y, get_r_i, get_v_i, get_alpha_i, loong_head_speed, L_loong_head, L_loong_body, plot_polar_scatter


def get_pos_of_corner(alpha_i: int, alpha_i_pre: int, r_i: int, r_i_pre: int, r_0: int, r_1: int, alpha_0: int, alpha_1: int):
    """
    获取三个点的坐标
    """

    # 孔距边界距离 m
    AD_d = 0.275
    # 板凳长度的一半
    CD_d = 0.15
    # A C的距离
    AC_d = sp.sqrt(AD_d ** 2 + CD_d ** 2)

    # 龙头两个把手的间距
    D_1 = L_loong_head
    # 龙身两个把手的间距
    D_2 = L_loong_body

    Gama = sp.atan(CD_d / AD_d)

    gama_i_2 = - Gama + alpha_i_pre - alpha_i + sp.asin(r_i / D_2 * sp.sin(alpha_i_pre - alpha_i))
    gama_i_1 = - Gama + alpha_i_pre - alpha_i + sp.asin(r_i_pre / D_2 * sp.sin(alpha_i_pre - alpha_i))
    gama_0 = Gama + alpha_0 - alpha_1 + sp.asin(r_1 / D_1 * sp.sin(alpha_0 - alpha_1))

    r_B_i = sp.sqrt(AC_d ** 2 + r_i ** 2 - 2 * AC_d * r_i * sp.cos(gama_i_1))
    r_B_i_pre = sp.sqrt(AC_d ** 2 + r_i_pre ** 2 - 2 * AC_d * r_i_pre * sp.cos(gama_i_2) )
    r_C_0 = sp.sqrt(AC_d ** 2 + r_0 ** 2 - 2 * AC_d * r_0 * sp.cos(gama_0) )

    beta_B_i = alpha_i_pre + sp.asin(AC_d / r_B_i * sp.sin(gama_i_1))
    beta_B_i_pre = alpha_i - sp.asin(AC_d / r_B_i_pre * sp.sin(gama_i_2))
    beta_C_0 = alpha_0 + sp.asin(AC_d / r_0 * sp.sin(gama_0))

    x_B_i, y_B_i = get_x_y(alpha=beta_B_i, r=r_B_i)
    x_B_i_pre, y_B_i_pre = get_x_y(alpha=beta_B_i_pre, r=r_B_i_pre)
    x_C_0, y_C_0 = get_x_y(alpha=beta_C_0, r=r_C_0)

    return (x_B_i, y_B_i), (x_B_i_pre, y_B_i_pre), (x_C_0, y_C_0)

def get_end_time():
    """
    获取龙头到达原点的时间 并将位置信息存入df_rav
    """
    t = sp.var("t")

    # 将 get_alpha_0 代入 get_r_0，得到关于 t 的函数
    alpha_0 = get_alpha_0(t)
    r_0 = get_r_0(alpha_0)
    # 求解方程 r_0 = 0
    equation = sp.Eq(r_0, 0)
    return float(sp.solve(equation, t)[0])


def forward_vaild(r_0: int, r_1: int, spiral_d: int):
    """
    @param r_0: 龙头极径
    @param r_1: 第一节龙身极径
    """
    detal_x = (r_0 * sp.cos(2 * np.pi * r_0 / spiral_d) - r_1 * sp.cos(2 * np.pi * r_1 / spiral_d))
    dx = sp.cos(2 * np.pi * r_0 / spiral_d - 2 * np.pi * r_0 / spiral_d * sp.sin(2 * np.pi * r_0 / spiral_d))
    detal_y = r_0 * sp.sin(2 * np.pi * r_0 / spiral_d) - r_1 * sp.sin(2 * np.pi * r_1 / spiral_d)
    dy = sp.sin(2 * np.pi * r_0 / spiral_d + 2 * np.pi * r_0 / spiral_d * sp.cos(2 * np.pi * r_0 / spiral_d))
    return  detal_x * dx + detal_y * dy >= 0




def are_collinear(A, B, C, atol=1e-4):
    """
    判断三个极坐标点 A、B、C 是否共线
    :param A: 点 A 的极坐标 (r_A, theta_A)
    :param B: 点 B 的极坐标 (r_B, theta_B)
    :param C: 点 C 的极坐标 (r_C, theta_C)
    :return: 如果 A、B、C 共线，返回 True
    """
    # 将极坐标转换为笛卡尔坐标
    x_A, y_A = get_x_y(r=A[0], alpha=A[1])
    x_B, y_B = get_x_y(r=B[0], alpha=B[1])
    x_C, y_C = get_x_y(r=C[0], alpha=C[1])
    
    # 计算向量 AB 和 AC
    vector_AB = (x_B - x_A, y_B - y_A)
    vector_AC = (x_C - x_A, y_C - y_A)
    
    # 计算向量 AB 和 AC 的叉积
    cross_product = vector_AB[0] * vector_AC[1] - vector_AB[1] * vector_AC[0]
    # 共线 且 A在BC之间
    return np.isclose(cross_product, 0, atol=atol) and (x_A - x_B) * (x_A - x_C) <= 0


def get_min_t(back_time=100, point_num=10):
    """
    第一次碰撞的时间
    """

    end = get_end_time()
    # print("到达零点时间", end)

    # 认为碰撞仅存在2-10 这些点
    time_range = np.arange(end - back_time, end, 0.001).tolist()
    df_rav = pd.DataFrame(index=pd.MultiIndex.from_product([range(10), ['r', 'alpha', "v"]]), columns=time_range)

    for t in time_range:
        print(t)
        for point_index in range(point_num):
            if point_index != 0:
                r_i_pre=df_rav[t][point_index - 1, 'r']
                v_i_pre=df_rav[t][point_index - 1, 'v']

                cur_r = get_r_i(r_i_pre=r_i_pre, point_index=point_index)
                cur_alpha = get_alpha_i(r_i=cur_r)

                cur_v = get_v_i(v_i_pre=v_i_pre, r_i_pre=r_i_pre, r_i=cur_r)
            else:
                cur_alpha = get_alpha_0(t)
                cur_r = get_r_0(cur_alpha)
                cur_v = loong_head_speed

            cur_x, cur_y = get_x_y(alpha=cur_alpha, r=cur_r)
            
            # 记录当前的 r 和 alpha
            df_rav.at[(point_index, 'r'), t] = cur_r
            df_rav.at[(point_index, 'alpha'), t] = cur_alpha
            df_rav.at[(point_index, 'v'), t] = cur_v


            if point_index > 4:
                alpha_0 = df_rav.at[(0, 'alpha'), t]
                r_0 = df_rav.at[(0, 'r'), t]

                alpha_1 = df_rav.at[(1, 'alpha'), t]
                r_1 = df_rav.at[(1, 'r'), t]

                alpha_i_pre = df_rav.at[(point_index - 1, 'alpha'), t]
                r_i_pre = df_rav.at[(point_index - 1, 'r'), t]

                B_i, B_i_pre, C_0 = get_pos_of_corner(alpha_0=alpha_0, r_0=r_0, alpha_1=alpha_1, r_1=r_1, alpha_i_pre=alpha_i_pre, r_i_pre=r_i_pre, alpha_i=cur_alpha, r_i=cur_r )
                if are_collinear(B_i, B_i_pre, C_0):
                    print(f'{t}时刻发现碰撞点 r_0为{r_0}')
                    return t
        if not forward_vaild(r_0=df_rav[t][0, 'r'], r_1=df_rav[t][1, 'r'], spiral_d=0.55):
            print(f'{t}时刻 r_0 r_1为 锐角')
            return t


if __name__ == "__main__":
    get_min_t()
                    





