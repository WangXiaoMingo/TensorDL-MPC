

def SimSISO1(noise_amplitude=1):
    import numpy as np
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from dynamics import SISO1
    import pandas as pd
    # 初始化系统状态
    initial_state = np.array([1, 1.2, 0.8])  # 初始状态可以自定义，y[0]，y[1]
    initial_input = np.array([0, 0.1, 0.2])  # 初始状态可以自定义，u[0],u[1]
    simulation = SISO1.SystemPlant(initial_state, initial_input, noise_amplitude, sine_wave=False)  # 信号生成方式
    return simulation


def SimSISO(noise_amplitude=1):
    import numpy as np
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from dynamics import SISO
    import pandas as pd
    # 初始化系统状态
    initial_state = np.array([1, 1.2])  # 初始状态可以自定义，y[0]，y[1]
    initial_input = np.array([0.1, 0.2])  # 初始状态可以自定义，u[0],u[1]
    simulation = SISO.SystemPlant(initial_state, initial_input, noise_amplitude, sine_wave=False)  # 信号生成方式
    return simulation

