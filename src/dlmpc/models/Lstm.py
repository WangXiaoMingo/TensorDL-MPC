# dlmpc/models/Lstm.py

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


class SeriesLstm:
    def __init__(self, lstm_blocks = 1):
        super().__init__()
        self.lstm_blocks =lstm_blocks

    def build(self, lstm_units=32,dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x,1))
        input_u = keras.Input(shape=(dim_u,1))
        if data_type == '1D':
            # inputs = tf.concat([input_x,input_u], axis=1)   # [y1,y2,x1,x2]
            inputs = tf.transpose(tf.concat([input_x, input_u], axis=1), [0, 2, 1])  # [y1,y2,x1,x2]
        elif data_type == '2D':
            if (dim_x+dim_u) % 2 == 0:
                inputs = tf.reshape(tf.concat([input_x,input_u], axis=1),[-1,int((dim_x+dim_u)/2),2])  # [[y1,y2],[x1,x2]]
        elif data_type == '2DT':
            inputs = tf.transpose(tf.reshape(tf.concat([input_x, input_u], axis=1), [-1, int((dim_x+dim_u)/2), 2]),[0,2,1])  # [[y1,x1],[y2,x2]]

        for j in range(self.lstm_blocks):
            if j == self.lstm_blocks - 1:
                x = keras.layers.LSTM(lstm_units, return_sequences=False)(inputs)
                x = keras.layers.LayerNormalization()(x)
            else:
                x = keras.layers.LSTM(lstm_units,return_sequences=True)(inputs)
                x = keras.layers.LayerNormalization()(x)
                inputs = x
        x = keras.layers.Dense(dense_units,activation='relu')(x)
        out = keras.layers.Dense(1,activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out,name='SeriesLstm')
        return model

class NormLstm:
    def __init__(self, lstm_blocks = 1):
        super().__init__()
        self.lstm_blocks =lstm_blocks

    def build(self, lstm_units=32,dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x,1))
        input_u = keras.Input(shape=(dim_u,1))
        if data_type == '1D':
            inputs = tf.concat([input_x,input_u], axis=1)   # [y1,y2,x1,x2]
        elif data_type == '2D':
            if (dim_x+dim_u) % 2 == 0:
                inputs = tf.reshape(tf.concat([input_x,input_u], axis=1),[-1,int((dim_x+dim_u)/2),2])  # [[y1,y2],[x1,x2]]
        elif data_type == '2DT':
            inputs = tf.transpose(tf.reshape(tf.concat([input_x, input_u], axis=1), [-1, int((dim_x+dim_u)/2), 2]),[0,2,1])  # [[y1,x1],[y2,x2]]

        for j in range(self.lstm_blocks):
            if j == self.lstm_blocks - 1:
                x = keras.layers.LSTM(lstm_units, return_sequences=False)(inputs)
                x = keras.layers.LayerNormalization()(x)
            else:
                x = keras.layers.LSTM(lstm_units,return_sequences=True)(inputs)
                x = keras.layers.LayerNormalization()(x)
                inputs = x
        x = keras.layers.Dense(dense_units,activation='relu')(x)
        out = keras.layers.Dense(1,activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out,name='NormLstm')
        return model

class LSTM:
    def __init__(self):
        super().__init__()

    def build(self, lstm_blocks=1, lstm_units=32, dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x,1))
        input_u = keras.Input(shape=(dim_u,1))
        if data_type == '1D':
            inputs = tf.concat([input_x,input_u], axis=1)   # [y1,y2,x1,x2]
        elif data_type == '2D':
            if (dim_x+dim_u) % 2 == 0:
                inputs = tf.reshape(tf.concat([input_x,input_u], axis=1),[-1,int((dim_x+dim_u)/2),2])  # [[y1,y2],[x1,x2]]
        elif data_type == '2DT':
            inputs = tf.transpose(tf.reshape(tf.concat([input_x, input_u], axis=1), [-1, int((dim_x+dim_u)/2), 2]),[0,2,1])  # [[y1,x1],[y2,x2]]

        for j in range(lstm_blocks):
            if j == lstm_blocks - 1:
                x = keras.layers.LSTM(lstm_units, return_sequences=False)(inputs)
            else:
                x = keras.layers.LSTM(lstm_units,return_sequences=True)(inputs)
            inputs = x
        x = keras.layers.Dense(dense_units)(x)
        out = keras.layers.Dense(1,activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out,name='LSTM')
        return model

class ResnetLstm:
    def __init__(self, lstm_blocks=3):
        super().__init__()
        self.lstm_blocks = lstm_blocks

    def build(self, lstm_units=32,dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x,1))
        input_u = keras.Input(shape=(dim_u,1))
        if data_type == '1D':
            inputs = tf.concat([input_x,input_u], axis=1)   # [y1,y2,x1,x2]
        elif data_type == '2D':
            if (dim_x+dim_u) % 2 == 0:
                inputs = tf.reshape(tf.concat([input_x,input_u], axis=1),[-1,int((dim_x+dim_u)/2),2])  # [[y1,y2],[x1,x2]]
        elif data_type == '2DT':
            inputs = tf.transpose(tf.reshape(tf.concat([input_x, input_u], axis=1), [-1, int((dim_x+dim_u)/2), 2]),[0,2,1])  # [[y1,x1],[y2,x2]]

        for j in range(self.lstm_blocks):
            if j == self.lstm_blocks - 1:
                x = keras.layers.LSTM(lstm_units, return_sequences=False)(inputs)
                x = keras.layers.LayerNormalization()(x)
                x = tf.concat([keras.layers.Flatten()(inputs), x], axis=-1)
            else:
                x = keras.layers.LSTM(lstm_units,return_sequences=True)(inputs)
                x = keras.layers.LayerNormalization()(x)
                x = tf.concat([inputs, x], axis=-1)
                inputs = x
        x = keras.layers.Dense(dense_units,activation='relu')(x)
        out = keras.layers.Dense(1,activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out, name='ResnetLstm')
        return model

class SkipLstm:
    def __init__(self, lstm_blocks=3):
        super().__init__()
        self.lstm_blocks = lstm_blocks

    def build(self, lstm_units=32, dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x, 1))
        input_u = keras.Input(shape=(dim_u, 1))
        if data_type == '1D':
            inputs = tf.concat([input_x, input_u], axis=1)  # [y1,y2,x1,x2]
        elif data_type == '2D':
            if (dim_x + dim_u) % 2 == 0:
                inputs = tf.reshape(tf.concat([input_x, input_u], axis=1),
                                    [-1, int((dim_x + dim_u) / 2), 2])  # [[y1,y2],[x1,x2]]
        elif data_type == '2DT':
            inputs = tf.transpose(tf.reshape(tf.concat([input_x, input_u], axis=1), [-1, int((dim_x + dim_u) / 2), 2]),
                                  [0, 2, 1])  # [[y1,x1],[y2,x2]]
        flat_inputs = keras.layers.Flatten()(inputs)

        for j in range(self.lstm_blocks):
            if j == self.lstm_blocks - 1:
                x = keras.layers.LSTM(lstm_units, return_sequences=False)(inputs)
                x = keras.layers.LayerNormalization()(x)
                x = tf.concat([flat_inputs, x], axis=-1)

            else:
                x = keras.layers.LSTM(lstm_units, return_sequences=True)(inputs)
                x = keras.layers.LayerNormalization()(x)
                x = tf.concat([inputs, x], axis=-1)
                inputs = x
        x = keras.layers.Dense(dense_units, activation='relu')(x)
        out = keras.layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out, name= 'SkipLstm')
        return model

class ResSkipLstm:
    def __init__(self, lstm_blocks=3):
        super().__init__()
        self.lstm_blocks = lstm_blocks

    def build(self, lstm_units=32, dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x, 1))
        input_u = keras.Input(shape=(dim_u, 1))
        if data_type == '1D':
            inputs = tf.concat([input_x, input_u], axis=1)  # [y1,y2,x1,x2]
        elif data_type == '2D':
            if (dim_x + dim_u) % 2 == 0:
                inputs = tf.reshape(tf.concat([input_x, input_u], axis=1),
                                    [-1, int((dim_x + dim_u) / 2), 2])  # [[y1,y2],[x1,x2]]
        elif data_type == '2DT':
            inputs = tf.transpose(tf.reshape(tf.concat([input_x, input_u], axis=1), [-1, int((dim_x + dim_u) / 2), 2]),
                                  [0, 2, 1])  # [[y1,x1],[y2,x2]]
        flat_inputs = keras.layers.Flatten()(inputs)

        for j in range(self.lstm_blocks):
            if j == self.lstm_blocks - 1:
                x = keras.layers.LSTM(lstm_units, return_sequences=False)(inputs)
                x = keras.layers.LayerNormalization()(x)
                if self.lstm_blocks == 1:
                    x = tf.concat([flat_inputs, x], axis=-1)
                else:
                    x = tf.concat([keras.layers.Flatten()(inputs), flat_inputs, x], axis=-1)
            else:
                x = keras.layers.LSTM(lstm_units, return_sequences=True)(inputs)
                x = keras.layers.LayerNormalization()(x)
                x = tf.concat([inputs, x], axis=-1)
                inputs = x
        x = keras.layers.Dense(dense_units, activation='relu')(x)
        out = keras.layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out, name='ResSkipLstm')
        return model


if __name__ == '__main__':
    model_net = SeriesLstm(lstm_blocks=1)
    model = model_net.build(lstm_units=32, dense_units=64, dim_u=1, dim_x=3, data_type='2DT')
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    print(model.name)

    model_net = NormLstm(lstm_blocks=1)
    model = model_net.build(lstm_units=32, dense_units=64, dim_u=1, dim_x=3, data_type='2DT')
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    print(model.name)


    model_net = LSTM()
    model = model_net.build(lstm_blocks=1, lstm_units=32, dense_units=64, dim_u=1, dim_x=3, data_type='2DT')
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    print(model.name)


    model_net = ResnetLstm(lstm_blocks=3)
    model = model_net.build(lstm_units=32, dense_units=64, dim_u=1, dim_x=3, data_type='2DT')
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    print(model.name)


    model_net = SkipLstm(lstm_blocks=3)
    model = model_net.build(lstm_units=32, dense_units=64, dim_u=1, dim_x=3, data_type='2DT')
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    print(model.name)


    model_net = ResSkipLstm(lstm_blocks=3)
    model = model_net.build(lstm_units=32, dense_units=64, dim_u=1, dim_x=3, data_type='2DT')
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    print(model.name)


