"""
@Author: blankxiao
@file: A_3.py
@Created: 2024-09-06 21:46
@Desc: A_3
"""
import numpy as np
import pandas as pd

from A_2 import get_end_time, get_pos_of_corner, are_collinear
from A_1 import get_alpha_0, get_alpha_i, get_r_0, get_r_i, get_v_i, get_x_y, loong_head_speed





def get_crash_r_0(spiral_d, back_time=20, point_num=10):
    """
    第一次碰撞的时间
    """

    end = get_end_time()
    # print("到达零点时间", end)

    # 认为碰撞仅存在2-10 这些点
    time_range = np.arange(end - back_time, end, 0.001).tolist()
    df_rav = pd.DataFrame(index=pd.MultiIndex.from_product([range(10), ['r', 'alpha', "v"]]), columns=time_range)

    for t in time_range:
        for point_index in range(point_num):
            if point_index != 0:
                r_i_pre=df_rav[t][point_index - 1, 'r']
                v_i_pre=df_rav[t][point_index - 1, 'v']

                cur_r = get_r_i(r_i_pre=r_i_pre, point_index=point_index, spiral_d=spiral_d)
                cur_alpha = get_alpha_i(r_i=cur_r, spiral_d=spiral_d)

                cur_v = get_v_i(v_i_pre=v_i_pre, r_i_pre=r_i_pre, r_i=cur_r, spiral_d=spiral_d)
            else:
                cur_alpha = get_alpha_0(t, spiral_d=spiral_d)
                cur_r = get_r_0(cur_alpha, spiral_d=spiral_d)
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
                if are_collinear(B_i, B_i_pre, C_0, atol=1e-1):
                    print(f'{t}时刻发现碰撞点')
                    return r_0




def get_min_spiral_distance():
    # 螺线变化 尝试的步长
    step = 0.2
    spiral_d = 0.55
    times = 10
    while times := times - 1:
        crash_r_0 = get_crash_r_0(spiral_d=spiral_d)
        print(f'{crash_r_0}m')
        print(f'{spiral_d}m')
        if crash_r_0 > 4.5:
            spiral_d += step
            step /= 2
        else:
            spiral_d -= step
            step /= 2


if __name__ == "__main__":
    get_min_spiral_distance()




