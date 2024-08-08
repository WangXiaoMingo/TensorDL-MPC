# dlmpc/models/Resnet_tcm.py

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

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import os
# 固定随机数种子
random_seed = 42
random.seed(random_seed)  # set random seed for python
np.random.seed(random_seed)  # set random seed for numpy
tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
os.environ['TF_DETERMINISTIC_OPS'] = '1' # set random seed for tensorflow-gpu


class ResnetTcm():
    def __init__(self, num_blocks=3, kernel_size=3, nb_filters = 64):
        super().__init__()
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.dilation_rate = [2**i for i in range(num_blocks)]

    def build(self,lstm_blocks=1,lstm_units=32,dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
        '''resnet_tcm'''
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x,1))
        input_u = keras.Input(shape=(dim_u,1))
        if data_type == '1D':
            inputs = tf.concat([input_x,input_u], axis=1)         # [y1,y2,x1,x2]
            # inputs = tf.transpose(tf.concat([input_x, input_u], axis=1),[0,2,1])# [y1,y2,x1,x2]
        elif data_type == '2D':
            if (dim_x+dim_u) % 2 == 0:
                inputs = tf.reshape(tf.concat([input_x,input_u], axis=1),[-1,int((dim_x+dim_u)/2),2])        # [[y1,y2],[x1,x2]]
        elif data_type == '2DT':
            inputs = tf.transpose(tf.reshape(tf.concat([input_x, input_u], axis=1), [-1, int((dim_x+dim_u)/2), 2]),[0,2,1])  # # [[y1,x1],[y2,x2]]

        # 初始化因果卷积层
        for i in range(self.num_blocks):
            x = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='causal', dilation_rate=self.dilation_rate[i])(inputs)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.LayerNormalization()(x)
            for j in range(lstm_blocks):
                x = keras.layers.LSTM(lstm_units, return_sequences=True)(x)
                x = keras.layers.LayerNormalization()(x)
            x = keras.layers.Dropout(0.5)(x)
            conv_inputs = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='causal',
                                dilation_rate=1)(inputs)
            x = tf.concat([conv_inputs, x], axis=-1)
            inputs = x
        x = keras.layers.Dense(dense_units)(x)
        out = keras.layers.Dense(1,activation='linear')(x)[:,-1]
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out,name='ResnetTcm')
        return model

if __name__ == '__main__':
    my_model = ResnetTcm()
    model = my_model.build()
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    print(model.name)
