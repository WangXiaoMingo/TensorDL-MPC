# dlmpc/train_models/train_model.py

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

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler
import tensorflow.keras.backend as K
import time
from pylab import *
import warnings
warnings.filterwarnings("ignore")


class TrainModel:
    def __init__(self,model,lr = 0.001,epoch=200):
        self.model = model
        self.lr = lr
        self.epoch = epoch
        self.model_name = model.name
        optimer = Adam(self.lr)  #
        self.model.compile(loss='mse', optimizer=optimer, metrics=['mae'])

    def scheduler(self,epoch):
        # 每隔100个epoch，学习率减小为原来的1/10
        if epoch % 100 == 0 and epoch != 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(self.model.optimizer.lr)

    def train_model(self, train_data,valid_data, show_loss=True):
        a = time.time()
        # 保存每次训练过程中的最佳的训练模型,有一次提升, 则覆盖一次.
        checkpoint = ModelCheckpoint(f'models_save/{self.model_name}.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='min')
        reduce_lr = LearningRateScheduler(self.scheduler)
        callbacks_list = [checkpoint, reduce_lr]
        # 训练模型
        history = self.model.fit([train_data['train_x_sequences'],train_data['train_u_sequences']], train_data['train_y_sequences'], epochs=self.epoch, batch_size=128, shuffle=False,
                  validation_data=([valid_data['valid_x_sequences'], valid_data['valid_u_sequences']], valid_data['valid_y_sequences']), callbacks=callbacks_list)
        # model.summary()
        b = time.time() - a
        print(f'>>>>>> run time:>>{b} s')
        del self.model
        # 迭代图像
        if show_loss:
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs_range = range(self.epoch)
            plt.plot(epochs_range, loss, label='Train Loss')
            plt.plot(epochs_range, val_loss, label='Test Loss')
            plt.legend(loc='upper right')
            plt.title(f'{self.model_name} Train and Val Loss ')
            plt.xlabel('iterations')
            plt.xlabel('loss mse')
            plt.grid(False)
            plt.show()

