# dlmpc/datasets/data_loader.py

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

class DataLoader:
    ''' data split'''
    def __init__(self, dataset):
        '''
        :param dataset:  a set of dataset,  states,inputs,outputs
        '''
        self.x_sequences = dataset[0]
        self.u_sequences = dataset[1]
        self.y_sequences = dataset[2]

    def load_data(self,split_seed=[0.8,0.2],data_dim=3):
        assert max(split_seed) <= 1, 'split_seed should be in [0,1]'
        #split_seed = [8,2],[8,1,1]
        train_data = {}
        test_data = {}
        if len(split_seed) == 2:
            # 训练模型
            num = int(split_seed[0] * len(self.y_sequences))
            train_x_sequences, train_u_sequences, train_y_sequences = self.x_sequences[:num, :, :], self.u_sequences[:num, :, :], self.y_sequences[:num, :, :]
            test_x_sequences, test_u_sequences, test_y_sequences = self.x_sequences[num:, :, :], self.u_sequences[num:, :,:], self.y_sequences[num:, :, :]
            train_data['train_x_sequences'] = train_x_sequences
            train_data['train_u_sequences'] = train_u_sequences
            train_data['train_y_sequences'] = train_y_sequences

            test_data['test_x_sequences'] = test_x_sequences
            test_data['test_u_sequences'] = test_u_sequences
            test_data['test_y_sequences'] = test_y_sequences
            return (train_data, test_data)

        elif len(split_seed) == 3:
            if data_dim == 3:
                valid_data = {}
                num = int(split_seed[0] * len(self.y_sequences))
                num1 = int((split_seed[0]+split_seed[1]) * len(self.y_sequences))
                train_x_sequences, train_u_sequences, train_y_sequences = self.x_sequences[:num, :, :], self.u_sequences[:num, :, :], self.y_sequences[:num, :, :]
                valid_x_sequences, valid_u_sequences, valid_y_sequences = self.x_sequences[num:num1, :, :], self.u_sequences[num:num1, :,:], self.y_sequences[num:num1, :, :]
                test_x_sequences, test_u_sequences, test_y_sequences = self.x_sequences[num:, :, :], self.u_sequences[num:,:,:], self.y_sequences[num:, :, :]
            elif data_dim == 2:
                valid_data = {}
                num = int(split_seed[0] * len(self.y_sequences))
                num1 = int((split_seed[0]+split_seed[1]) * len(self.y_sequences))
                train_x_sequences, train_u_sequences, train_y_sequences = self.x_sequences[:num, :], self.u_sequences[:num, :], self.y_sequences[:num, :]
                valid_x_sequences, valid_u_sequences, valid_y_sequences = self.x_sequences[num:num1, :], self.u_sequences[num:num1, :], self.y_sequences[num:num1, :]
                test_x_sequences, test_u_sequences, test_y_sequences = self.x_sequences[num:, :], self.u_sequences[num:,:], self.y_sequences[num:, :]

            train_data['train_x_sequences'] = train_x_sequences
            train_data['train_u_sequences'] = train_u_sequences
            train_data['train_y_sequences'] = train_y_sequences

            test_data['test_x_sequences'] = test_x_sequences
            test_data['test_u_sequences'] = test_u_sequences
            test_data['test_y_sequences'] = test_y_sequences

            valid_data['valid_x_sequences'] = valid_x_sequences
            valid_data['valid_u_sequences'] = valid_u_sequences
            valid_data['valid_y_sequences'] = valid_y_sequences
            return (train_data, valid_data, test_data)