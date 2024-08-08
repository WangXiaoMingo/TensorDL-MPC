# dlmpc/datasets/preprocess.py

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


class WindowGenerator:
    def __init__(self, input_window_dy, input_window_du, u, y, u_dim=1):
        '''
        系统表达式： y[k] = f(y[k-1],y[k-2],……y[k-dy],u[k-1],u[k-2],……,u[k-du])
        令系统状态为：x =  [y[k-1],y[k-2],……y[k-dy],u[k-2],……,u[k-du]]
        模型控制输入为： u = u[k-1]
        因此，模型表达式为： y[k] = f(x,u)
        :param input_window_dy: 系统状态，过去时刻
        :param input_window_du: 系统状态，过去时刻
        :param data_u: 输入控制量u
        :param data_y: 系统输出y
        :param u: 输入变量个数
        '''
        self.window_dy = input_window_dy
        self.window_du = input_window_du
        self.data_u = u
        self.data_y = y
        self.u_dim = u_dim
        self.state_num = input_window_dy + input_window_du - 1    # 系统状态数目
        self.start_seq = max(self.window_dy,self.window_du)       # 开始标号
        self.num_samples = len(y) - self.start_seq                # 样本个数
        self.x = np.zeros((self.num_samples,self.state_num))      # 系统状态
        self.u = np.zeros((self.num_samples, self.u_dim))         # 系统控制量
        self.y = np.zeros((self.num_samples, 1))

    def generate_sequences(self):
        for i in range(self.start_seq,len(self.data_y)):
            # 生成状态数据窗口,self.x由两部分构成
            temp_y_state = self.data_y[i - self.window_dy : i]
            temp_u_state = self.data_u[i - self.window_du : i - 1]
            temp_state = np.concatenate((temp_y_state,temp_u_state),axis=0)  # 拼接得到状态x
            self.x[i - self.start_seq] = temp_state
            # 生成控制输入窗口
            self.u[i - self.start_seq] = self.data_u[i - 1]
            # 生成输出窗口
            self.y[i - self.start_seq] = self.data_y[i]
        return self.x, self.u, self.y

    def generate_2D_sequences(self):
        self.x, self.u, self.y = self.generate_sequences()
        return self.x, self.u, self.y

    def generate_3D_sequences(self):
        self.x, self.u, self.y = self.generate_sequences()
        return np.atleast_3d(self.x),np.atleast_3d(self.u),np.atleast_3d(self.y)

# 使用类
if __name__ == '__main__':
    # 示例数据
    u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 系统输出
    y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # 系统输出
    # 窗口大小
    input_window_dy = 2
    input_window_du = 2
    # 创建窗口生成器
    window_generator = WindowGenerator(input_window_dy, input_window_du, u, y, u_dim=1)
    # 生成序列
    # x_sequences, u_sequences, y_sequences = window_generator.generate_2D_sequences()
    x_sequences, u_sequences, y_sequences = window_generator.generate_3D_sequences()
    # 打印结果
    print("X sequences:")
    print(x_sequences,x_sequences.shape)
    print("U sequences:")
    print(u_sequences,u_sequences.shape)
    print("Y sequences:")
    print(y_sequences,y_sequences.shape)
