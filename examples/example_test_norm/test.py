import numpy as np
from scipy import signal

# 定义传递函数参数
num = [3]  # 分子系数
den = [250, 35, 1]  # 分母系数

# 创建传递函数模型
system = signal.TransferFunction(num, den)

# 定义采样时间
Ts = 1  # 采样时间，单位秒

# 将传递函数转换为离散时间模型
system_discrete = signal.cont2discrete((num, den), Ts, method='tustin')

# 定义输入信号u，这里我们假设它是一个序列
# 假设我们有一个MPC控制器，它为每个时间步计算出一个输入u
# 这里我们用一个简单的序列来模拟MPC的输出
u = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 示例输入序列

# 初始化输出序列
y = np.zeros_like(u)

# 初始化系统状态
x = np.zeros((len(den) - 1,))  # 状态向量初始化为0

# 定义传递函数参数
num = [3]  # 分子系数
den = [250, 35, 1]  # 分母系数
delay = 20  # 时间延迟，单位秒

# 创建带有时间延迟的传递函数模型
system = signal.TransferFunction(num, den)
print(system)

print(signal.lsim(system,U=[1,1,1,1],T=[1,2,3,4]))
