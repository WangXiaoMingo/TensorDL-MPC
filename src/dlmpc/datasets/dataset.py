# dlmpc/datasets/dataset.py

# Software Copyright Notice

#   This file is part of DL-MPC

#   DL-MPC: A toolbox for deep learning-based nonlinear model predictive control

#   GNU Affero General Public License version 3.0
#   Copyright (c) 2024, Xiaoming Wang. All rights reserved

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.

#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https:#www.gnu.org/licenses/>.

#   This software may contain third-party software components, the use and distribution of which are subject to the
#   respective third-party license agreements. If you need to use third-party software components,
#   please ensure compliance with their license agreements.

#   If you have any questions about the software or require further assistance,
#   please contact Xiaoming Wang support team. e-mail: wangxiaoming19951@163.com

#   Last updated on June, 2024
#   Author: Xiaoming Wang


import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, plant_name='SISO',noise_amplitude=1):
        if plant_name == 'SISO':
            from ..dynamics import SISO
            initial_state = np.array([1, 1.2])  # 初始状态可以自定义，y[0]，y[1]
            initial_input = np.array([0.1, 0.2])  # 初始状态可以自定义，u[0],u[1]
            self.simulation = SISO.SystemPlant(initial_state, initial_input, noise_amplitude, sine_wave=False)  # 信号生成方式
        elif plant_name == 'SISO1':
            from ..dynamics import SISO1
            initial_state = np.array([1, 1.2, 0.8])  # 初始状态可以自定义，y[0]，y[1]
            initial_input = np.array([0, 0.1, 0.2])  # 初始状态可以自定义，u[0],u[1]
            self.simulation = SISO1.SystemPlant(initial_state, initial_input, noise_amplitude, sine_wave=False)  # 信号生成方式

    def preprocess(self,num=1000):
        y, u = self.simulation.generate_data(num)
        self.simulation.plot_results()
        data = pd.DataFrame(u, index=[f'k_{i}' for i in range(len(u))], columns=['u'])
        data['y'] = y
        print(data)
        return data
