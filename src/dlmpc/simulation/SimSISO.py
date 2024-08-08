
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dynamics import SISO
import pandas as pd
# 初始化系统状态
initial_state = np.array([1, 1.2])  # 初始状态可以自定义，y[0]，y[1]
initial_input = np.array([0.1, 0.2])  # 初始状态可以自定义，u[0],u[1]
simulation = SISO.SystemPlant(initial_state, initial_input, noise_amplitude=1, sine_wave=False)  # 信号生成方式
# y, u = simulation.generate_data(1000)
# simulation.plot_results()
# data = pd.DataFrame(u, index=[f'k_{i}' for i in range(len(u))], columns=['u'])
# data['y'] = y