import numpy as np
import matplotlib.pyplot as plt

# 设置参数
omega = 2 * np.pi * 5e9  # 角频率，5 GHz
k = omega / 3e8  # 波数，假设在真空中
x = np.linspace(-1, 1, 1000)  # 空间位置，单位可以是米
t = np.linspace(0, 1e-9, 1000)  # 时间，单位秒


def electric_field(x, t, amplitude=1, phase_shift=0):
    # 电场随时间和空间的变化
    E0 = amplitude  # 电场振幅
    Ex = E0 * np.cos(k * x - omega * t + phase_shift)  # 电场沿y轴
    return Ex


# 假设相位偏移为0
phase_shift = 0

# 计算特定时间点的电场分布
t_fixed = t[500]  # 选择一个时间点
Ey = electric_field(x, t_fixed, phase_shift=phase_shift)

# 绘制电场分布
plt.figure(figsize=(8, 4))
plt.plot(x, Ey, label='Ey at t = {:.2e} s'.format(t_fixed))
plt.xlabel('x (m)')
plt.ylabel('Ey (V/m)')
plt.title('Electric Field Distribution')
plt.legend()
plt.grid(True)
plt.show()