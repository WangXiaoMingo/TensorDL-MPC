# dlmpc/layers/normlization.py

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

class MinMaxNormalization(tf.keras.layers.Layer):
    def __init__(self, feature_range=(0, 1), min_val=None, max_val=None, name=None, **kwargs):
        super(MinMaxNormalization, self).__init__(name=name, **kwargs)
        self.feature_range = feature_range
        self.min_val = min_val
        self.max_val = max_val
        self._min_val_var = None
        self._max_val_var = None

    def build(self, input_shape):
        # 将 min_val 和 max_val 初始化为变量
        if self.min_val is None:
            self._min_val_var = self.add_weight(
                shape=(),
                initializer='zeros',
                trainable=False,
                name=f'{self.name}_min'
            )
        else:
            self._min_val_var = tf.Variable(self.min_val, trainable=False, name=f'{self.name}_min')

        if self.max_val is None:
            self._max_val_var = self.add_weight(
                shape=(),
                initializer='ones',
                trainable=False,
                name=f'{self.name}_max'
            )
        else:
            self._max_val_var = tf.Variable(self.max_val, trainable=False, name=f'{self.name}_max')

        super(MinMaxNormalization, self).build(input_shape)

    def call(self, inputs):
        norm_inputs = (inputs - self._min_val_var) / (self._max_val_var - self._min_val_var)
        return norm_inputs * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def inverse_call(self, inputs):
        # 执行反归一化操作
        original_inputs = (inputs - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return original_inputs * (self._max_val_var - self._min_val_var) + self._min_val_var


    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(MinMaxNormalization, self).get_config()
        config.update({
            'feature_range': self.feature_range,
            'min_val': self._min_val_var.numpy(),  # 使用 numpy 方法来获取变量的值
            'max_val': self._max_val_var.numpy()
        })
        return config

    def update_min_max(self, inputs):
        inputs_tensor = tf.convert_to_tensor(inputs)
        new_min = tf.reduce_min(inputs_tensor)
        new_max = tf.reduce_max(inputs_tensor)

        # 确保 self._min_val_var 和 self._max_val_var 是 tf.Variable
        if not isinstance(self._min_val_var, tf.Variable):
            self._min_val_var.assign(new_min)
        else:
            self._min_val_var = tf.Variable(new_min, trainable=False, name=f'{self.name}_min')

        if not isinstance(self._max_val_var, tf.Variable):
            self._max_val_var.assign(new_max)
        else:
            self._max_val_var = tf.Variable(new_max, trainable=False, name=f'{self.name}_max')


if __name__ == '__main__':

    from sklearn import preprocessing
    import numpy as np

    #新的测试数据进来，同样的转换
    x = np.array([[-3,-1,-1],
     [1,1,10]])

    x_test = np.array([[1.,-1.,2.],
     [2.,0.,0.],
     [0.,1.,-1.]])
    min_max_scaler = preprocessing.MinMaxScaler()#默认为范围0~1，拷贝操作
    #min_max_scaler = preprocessing.MinMaxScaler(feature_range = (1,3),copy = False)#范围改为1~3，对原数组操作
    x_minmax = min_max_scaler.fit_transform(x)
    print('x_minmax = ',x_minmax)
    print('x = ',x)


    x_test_maxabs = min_max_scaler.transform(x_test)
    print('x_test_maxabs = ',x_test_maxabs)


    # 使用自定义层，并手动指定最小值和最大值
    min_val = tf.constant([-3., -1., -1.])  # 手动指定的最小值
    max_val = tf.constant([1, 1., 10.])   # 手动指定的最大值
    normalization_layer = MinMaxNormalization(feature_range=(0, 1), min_val=min_val, max_val=max_val)
    b = normalization_layer(x_test)
    print('test1:', b)
    print('test_inv:', normalization_layer.inverse_call(b))
    print('oral',x_test)

    x = tf.constant(x,dtype=tf.float32)
    normalization_layer = MinMaxNormalization(feature_range=(0, 1))
    normalization_layer.build(x.shape)
    normalization_layer.update_min_max(x)
    print('test2:', normalization_layer(x_test))


    a= tf.random.normal((3,3,3))
    print(a)
    normalization_layer = MinMaxNormalization(feature_range=(0, 1), min_val=min_val, max_val=max_val)
    x_test_maxabs = min_max_scaler.transform(a[0])
    print('x_test_maxabs1 = ',x_test_maxabs)

    x_test_maxabs = min_max_scaler.transform(a[1])
    print('x_test_maxabs2 = ',x_test_maxabs)

    x_test_maxabs = min_max_scaler.transform(a[2])
    print('x_test_maxabs3 = ',x_test_maxabs)
    print(normalization_layer(a))
