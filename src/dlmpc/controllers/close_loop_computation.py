# dlmpc/controllers/close_loop_computation.py

# Software Copyright Notice

#   This file is part of DL-MPC
#
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

import random
import time
import numpy as np
import tensorflow as tf

class OnlineOptimize:
    def __init__(self,mpc_controller,state_y, state_u,use_online_correction=True):
        self.state_y = state_y
        self.state_u = state_u
        self.mpc_controller = mpc_controller
        self.control_horizon = mpc_controller.control_horizon
        self.ly = mpc_controller.ly
        self.lu = mpc_controller.lu
        self.use_online_correction = use_online_correction


    def make_step(self,error, y_ref, iterations=100, tol=1e-6,function_type='du'):
        # 优化控制输入

        t_start = time.time()
        J, u_sequence, epoch = self.mpc_controller.optimize(error, self.state_y, self.state_u, y_ref, iterations,tol,function_type)
        solving_time = time.time() - t_start
        # 实施第一个控制输入
        u0 = u_sequence[0]
        # print(u_sequence)
        return {'u0':u0, 'solving_time':solving_time,'solving_J':J[0][0], 'solving_epoch':epoch}

    def estimate(self, u0, plant_out):
        # 更新状态
        model = self.mpc_controller.model
        x_current = np.concatenate((self.state_y, self.state_u), axis=0).reshape(1, self.ly+self.lu, 1)
        # print([x_current, np.atleast_3d(u0)])
        y_nn = model.predict([x_current, np.atleast_3d(u0)], verbose=0)
        # print(y_nn)
        # 更新状态
        state_y = tf.roll(self.state_y, -1, axis=0)
        state_u = tf.roll(self.state_u, -1, axis=0)
        state_y = tf.tensor_scatter_nd_update(state_y, [[self.ly - 1, 0]], plant_out)
        state_u = tf.tensor_scatter_nd_update(state_u, [[self.lu - 1, 0]], u0)

        if self.use_online_correction:
            error = plant_out[0] - y_nn[0]
        else:
            error = 0
        return state_y, state_u, error

    def estimate_simu(self, u0):
        # 更新状态
        model = self.mpc_controller.model
        x_current = np.concatenate((self.state_y, self.state_u), axis=0).reshape(1, self.ly+self.lu, 1)
        error = np.random.random()*0.5
        y_nn = model.predict([x_current, np.atleast_3d(u0)], verbose=0) + error
        # 更新状态
        state_y = tf.roll(self.state_y, -1, axis=0)
        state_u = tf.roll(self.state_u, -1, axis=0)
        state_y = tf.tensor_scatter_nd_update(state_y, [[self.ly - 1, 0]], y_nn[0])
        state_u = tf.tensor_scatter_nd_update(state_u, [[self.lu - 1, 0]], u0)

        # a = 1 - (y_ref - y_plant)
        # a = np.where(a > 50, 50, np.where(a < 0.01, 0.01, a))
        # R = np.eye(control_horizon) * a * np.max(Q)
        return state_y, state_u, error,y_nn[0]