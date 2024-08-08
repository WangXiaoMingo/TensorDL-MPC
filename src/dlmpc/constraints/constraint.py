# dlmpc/constraint/mconstraint.py

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

class NonNegative(tf.keras.constraints.Constraint):

 def __call__(self, w):
   return w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)


class BoundedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, u):
        # 确保 u 的值在 lower_bound 和 upper_bound 之间
        return tf.clip_by_value(u, self.lower_bound, self.upper_bound)

    def get_config(self):
        return {
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound
        }

if __name__ == '__main__':
    weight = tf.constant((-1.0, 1.0))
    print(NonNegative()(weight))
    # tf.Tensor([-0.  1.], shape=(2,), dtype=float32)

    tf.keras.layers.Dense(4, kernel_constraint=NonNegative())

    u_bounds = [-0.5, 0.5]
    u = tf.Variable(tf.random.uniform((5,1)),dtype=tf.float32)
    bounded_constraint = BoundedConstraint(lower_bound=u_bounds[0], upper_bound=u_bounds[1])
    print(bounded_constraint(u))
    #tf.Tensor([[0.5       ]
     # [0.5       ]
     # [0.33824527]
     # [0.5       ]
     # [0.240731  ]], shape=(5, 1), dtype=float32)






