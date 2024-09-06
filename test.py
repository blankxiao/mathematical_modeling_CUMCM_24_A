import sympy as sp

# 定义符号变量
r_i = sp.symbols('r_i')



def equation(r_i):
    # 假设 L_loong_body 和 L_loong_head 是已定义的常量
    bandeng_len = 1.6500000000000001
    spiral_d = 0.55

    # 假设 r_i_pre 是已定义的变量
    r_i_pre = 8.828527611357744
    return r_i_pre ** 2 + r_i ** 2 - 2 * r_i_pre * r_i * sp.cos(2 * sp.pi / spiral_d * (r_i - r_i_pre)) - bandeng_len ** 2

r_i_pre = 8.828527611357744

# 求解方程
eq = equation(r_i)
r_i_solution = sp.nsolve(eq, r_i, r_i_pre)  # 使用 nsolve 求解，初始猜测值为 1.0

print("Solution for r_i:", r_i_solution)