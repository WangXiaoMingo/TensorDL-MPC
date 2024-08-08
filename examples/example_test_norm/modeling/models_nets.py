import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
from get_windows_data import WindowGenerator
import random
import os


# 固定随机数种子
random_seed = 42
random.seed(random_seed)  # set random seed for python
np.random.seed(random_seed)  # set random seed for numpy
tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
os.environ['TF_DETERMINISTIC_OPS'] = '1' # set random seed for tensorflow-gpu
# warnings.filterwarnings("ignore")



class NNet():
    def __init__(self, num_blocks=3, kernel_size=3, nb_filters = 64):
        super().__init__()
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.dilation_rate = [2**i for i in range(num_blocks)]

    def resnet_tcm(self,lstm_blocks=1,lstm_units=32,dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
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
            x = tfa.layers.WeightNormalization(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='causal', dilation_rate=self.dilation_rate[i]))(inputs)
            x = keras.layers.Dropout(0.1)(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.LayerNormalization()(x)
            for j in range(lstm_blocks):
                x = keras.layers.LSTM(lstm_units, return_sequences=True)(x)
                x = keras.layers.LayerNormalization()(x)
            conv_inputs = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='causal',
                                dilation_rate=1)(inputs)
            x = keras.layers.Dropout(0.1)(x)
            x = tf.concat([conv_inputs, x], axis=-1)
            inputs = x
        x = keras.layers.Dense(dense_units,activation='relu')(x)
        out = keras.layers.Dense(1,activation='linear')(x)[:,-1]
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out)
        return model


    def skip_tcm(self,lstm_blocks=1,lstm_units=32,dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x,1))
        input_u = keras.Input(shape=(dim_u,1))
        if data_type == '1D':
            inputs = tf.concat([input_x,input_u], axis=1)   # [y1,y2,x1,x2]
            # inputs = tf.transpose(tf.concat([input_x, input_u], axis=1),[0,2,1])# [y1,y2,x1,x2]
        elif data_type == '2D':
            if (dim_x+dim_u) % 2 == 0:
                inputs = tf.reshape(tf.concat([input_x,input_u], axis=1),[-1,int((dim_x+dim_u)/2),2])  # [[y1,y2],[x1,x2]]
        elif data_type == '2DT':
            inputs = tf.transpose(tf.reshape(tf.concat([input_x, input_u], axis=1), [-1, int((dim_x+dim_u)/2), 2]),[0,2,1])  # [[y1,x1],[y2,x2]]

        # 初始化因果卷积层
        for i in range(self.num_blocks):
            x = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='causal', dilation_rate=self.dilation_rate[i])(inputs)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.LayerNormalization()(x)
            if i != self.num_blocks - 1:
                for j in range(lstm_blocks):
                    x = keras.layers.LSTM(lstm_units,return_sequences=True)(x)
                    x = keras.layers.LayerNormalization()(x)
                inputs = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='causal',
                                        dilation_rate=1)(inputs)
                x = keras.layers.Dropout(0.1)(x)
                x = tf.concat([inputs, x], axis=-1)

            else:
                for j in range(lstm_blocks-1):
                    x = keras.layers.LSTM(lstm_units, return_sequences=True)(x)
                    x = keras.layers.LayerNormalization()(x)
                x = keras.layers.LSTM(lstm_units, return_sequences=False)(x)
                x = keras.layers.LayerNormalization()(x)
                x = keras.layers.Dropout(0.1)(x)
                x = tf.concat([keras.layers.Flatten()(inputs), x], axis=-1)
            inputs = x
        x = keras.layers.Dense(dense_units,activation='relu')(x)
        out = keras.layers.Dense(1,activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out,name='SkipTcm')
        return model

    def resnet_skip_tcm(self,lstm_blocks=1,lstm_units=32,dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x,1))
        input_u = keras.Input(shape=(dim_u,1))
        if data_type == '1D':
            inputs = tf.concat([input_x,input_u], axis=1)   # [y1,y2,x1,x2]
            # inputs = tf.transpose(tf.concat([input_x, input_u], axis=1), [0, 2, 1])  # [y1,y2,x1,x2]
        elif data_type == '2D':
            if (dim_x+dim_u) % 2 == 0:
                inputs = tf.reshape(tf.concat([input_x,input_u], axis=1),[-1,int((dim_x+dim_u)/2),2])  # [[y1,y2],[x1,x2]]
        elif data_type == '2DT':
            inputs = tf.transpose(tf.reshape(tf.concat([input_x, input_u], axis=1), [-1, int((dim_x+dim_u)/2), 2]),[0,2,1])  # [[y1,x1],[y2,x2]]

        flat_inputs = keras.layers.Flatten()(inputs)
        # 初始化因果卷积层
        for i in range(self.num_blocks):
            x = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='causal', dilation_rate=self.dilation_rate[i])(inputs)
            x = keras.layers.Dropout(0.1)(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.LayerNormalization()(x)
            if i != self.num_blocks - 1:
                for j in range(lstm_blocks):
                    x = keras.layers.LSTM(lstm_units,return_sequences=True)(x)
                    x = keras.layers.LayerNormalization()(x)
                inputs = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='causal',
                                        dilation_rate=1)(inputs)
                x = keras.layers.Dropout(0.1)(x)
                x = tf.concat([inputs, x], axis=-1)
            else:
                for j in range(lstm_blocks-1):
                    x = keras.layers.LSTM(lstm_units, return_sequences=True)(x)
                    x = keras.layers.LayerNormalization()(x)
                x = keras.layers.LSTM(lstm_units, return_sequences=False)(x)
                x = keras.layers.LayerNormalization()(x)
                x = keras.layers.Dropout(0.1)(x)
                x = tf.concat([flat_inputs, x], axis=-1)
            inputs = x
        x = keras.layers.Dense(dense_units,activation='relu')(x)
        out = keras.layers.Dense(1,activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out)
        return model

    def resnet_2skip_tcm(self,lstm_blocks=1,lstm_units=32,dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
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

        flat_inputs = keras.layers.Flatten()(inputs)
        # 初始化因果卷积层
        for i in range(self.num_blocks):
            x = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='causal', dilation_rate=self.dilation_rate[i])(inputs)
            x = keras.layers.Dropout(0.1)(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.LayerNormalization()(x)
            if i != self.num_blocks - 1:
                for j in range(lstm_blocks):
                    x = keras.layers.LSTM(lstm_units,return_sequences=True)(x)
                    x = keras.layers.LayerNormalization()(x)
                inputs = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='causal',
                                        dilation_rate=1)(inputs)
                x = keras.layers.Dropout(0.1)(x)
                x = tf.concat([inputs, x], axis=-1)
            else:
                for j in range(lstm_blocks-1):
                    x = keras.layers.LSTM(lstm_units, return_sequences=True)(x)
                    x = keras.layers.LayerNormalization()(x)
                x = keras.layers.LSTM(lstm_units, return_sequences=False)(x)
                x = keras.layers.LayerNormalization()(x)
                x = keras.layers.Dropout(0.1)(x)
                x = tf.concat([keras.layers.Flatten()(inputs),flat_inputs, x], axis=-1)
            inputs = x
        x = keras.layers.Dense(dense_units,activation='relu')(x)
        out = keras.layers.Dense(1,activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out)
        return model

    def resnet_tcm_skip(self,lstm_blocks=1,lstm_units=32,dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
        '''resnet_tcm'''
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x,1))
        input_u = keras.Input(shape=(dim_u,1))
        if data_type == '1D':
            inputs = tf.concat([input_x,input_u], axis=1)         # [y1,y2,x1,x2]
        elif data_type == '2D':
            if (dim_x+dim_u) % 2 == 0:
                inputs = tf.reshape(tf.concat([input_x,input_u], axis=1),[-1,int((dim_x+dim_u)/2),2])        # [[y1,y2],[x1,x2]]
        elif data_type == '2DT':
            inputs = tf.transpose(tf.reshape(tf.concat([input_x, input_u], axis=1), [-1, int((dim_x+dim_u)/2), 2]),[0,2,1])  # # [[y1,x1],[y2,x2]]
        flat_inputs = keras.layers.Flatten()(inputs)
        # 初始化因果卷积层
        for i in range(self.num_blocks):
            x = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='causal', dilation_rate=self.dilation_rate[i])(inputs)
            # x = tfa.layers.WeightNormalization(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='same', dilation_rate=self.dilation_rate[i]))(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.LayerNormalization()(x)
            for j in range(lstm_blocks):
                x = keras.layers.LSTM(lstm_units, return_sequences=True)(x)
                x = keras.layers.LayerNormalization()(x)
            conv_inputs = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='causal',
                                dilation_rate=1)(inputs)
            x = keras.layers.Dropout(0.1)(x)
            x = tf.concat([conv_inputs, x], axis=-1)
            inputs = x
        x = tf.concat([keras.layers.Flatten()(x), flat_inputs], axis=-1)
        x = keras.layers.Dense(dense_units,activation='relu')(x)
        out = keras.layers.Dense(1,activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out)
        return model


    def resnet_tcm_series(self,lstm_blocks=1,lstm_units=32,dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
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

        # 初始化因果卷积层
        for i in range(self.num_blocks):
            x = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='causal', dilation_rate=self.dilation_rate[i])(inputs)
            # x = tfa.layers.WeightNormalization(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='same', dilation_rate=self.dilation_rate[i]))(inputs)
            x = keras.layers.Dropout(0.1)(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.LayerNormalization()(x)
            for j in range(lstm_blocks):
                if i == self.num_blocks-1 and j == lstm_blocks - 1:
                    x = keras.layers.LSTM(lstm_units, return_sequences=False)(x)
                    x = keras.layers.LayerNormalization()(x)
                else:
                    x = keras.layers.LSTM(lstm_units,return_sequences=True)(x)
                    x = keras.layers.LayerNormalization()(x)
                    inputs = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='causal',
                                            dilation_rate=1)(inputs)
                    x = tf.concat([inputs, x], axis=-1)
            inputs = x
        x = keras.layers.Dense(dense_units,activation='relu')(x)
        out = keras.layers.Dense(1,activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out)
        return model

    def series_tcm(self,lstm_blocks=1,lstm_units=32,dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
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

        # 初始化因果卷积层
        for i in range(self.num_blocks):
            x = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='same', dilation_rate=self.dilation_rate[i])(inputs)
            # x = tfa.layers.WeightNormalization(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='same', dilation_rate=self.dilation_rate[i]))(inputs)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.LayerNormalization()(x)
            for j in range(lstm_blocks):
                if i == self.num_blocks-1 and j == lstm_blocks - 1:
                    x = keras.layers.LSTM(lstm_units, return_sequences=False)(x)
                    x = keras.layers.LayerNormalization()(x)
                else:
                    x = keras.layers.LSTM(lstm_units,return_sequences=True)(x)
                    x = keras.layers.LayerNormalization()(x)
                    # x = keras.layers.Dense(dense_units, activation='relu')(x)
            inputs = x
        x = keras.layers.Dense(dense_units,activation='relu')(x)
        out = keras.layers.Dense(1,activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out)
        return model


    def series_lstm(self,lstm_blocks=1,lstm_units=32,dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
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
                x = keras.layers.LayerNormalization()(x)
            else:
                x = keras.layers.LSTM(lstm_units,return_sequences=True)(inputs)
                x = keras.layers.LayerNormalization()(x)
                inputs = x
        x = keras.layers.Dense(dense_units,activation='relu')(x)
        out = keras.layers.Dense(1,activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out)
        return model


    def resnet_lstm(self,lstm_blocks=1,lstm_units=32,dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
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
                x = keras.layers.LayerNormalization()(x)
                x = tf.concat([keras.layers.Flatten()(inputs), x], axis=-1)
            else:
                x = keras.layers.LSTM(lstm_units,return_sequences=True)(inputs)
                x = keras.layers.LayerNormalization()(x)
                x = tf.concat([inputs, x], axis=-1)
                inputs = x
        x = keras.layers.Dense(dense_units,activation='relu')(x)
        out = keras.layers.Dense(1,activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out)
        return model

    def skip_lstm(self, lstm_blocks=1, lstm_units=32, dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x, 1))
        input_u = keras.Input(shape=(dim_u, 1))
        if data_type == '1D':
            inputs = tf.concat([input_x, input_u], axis=1)  # [y1,y2,x1,x2]
            # inputs = tf.transpose(tf.concat([input_x, input_u], axis=1),[0,2,1])
        elif data_type == '2D':
            if (dim_x + dim_u) % 2 == 0:
                inputs = tf.reshape(tf.concat([input_x, input_u], axis=1),
                                    [-1, int((dim_x + dim_u) / 2), 2])  # [[y1,y2],[x1,x2]]
        elif data_type == '2DT':
            inputs = tf.transpose(tf.reshape(tf.concat([input_x, input_u], axis=1), [-1, int((dim_x + dim_u) / 2), 2]),
                                  [0, 2, 1])  # [[y1,x1],[y2,x2]]
        flat_inputs = keras.layers.Flatten()(inputs)

        for j in range(lstm_blocks):
            if j == lstm_blocks - 1:
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
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out)
        return model

    def resnet_skip_lstm(self, lstm_blocks=1, lstm_units=32, dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x, 1))
        input_u = keras.Input(shape=(dim_u, 1))
        if data_type == '1D':
            inputs = tf.concat([input_x, input_u], axis=1)  # [y1,y2,x1,x2]
            # inputs = tf.transpose(tf.concat([input_x, input_u], axis=1),[0,2,1])
        elif data_type == '2D':
            if (dim_x + dim_u) % 2 == 0:
                inputs = tf.reshape(tf.concat([input_x, input_u], axis=1),
                                    [-1, int((dim_x + dim_u) / 2), 2])  # [[y1,y2],[x1,x2]]
        elif data_type == '2DT':
            inputs = tf.transpose(tf.reshape(tf.concat([input_x, input_u], axis=1), [-1, int((dim_x + dim_u) / 2), 2]),
                                  [0, 2, 1])  # [[y1,x1],[y2,x2]]
        flat_inputs = keras.layers.Flatten()(inputs)

        for j in range(lstm_blocks):
            if j == lstm_blocks - 1:
                x = keras.layers.LSTM(lstm_units, return_sequences=False)(inputs)
                x = keras.layers.LayerNormalization()(x)
                if lstm_blocks == 1:
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
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out)
        return model

    def stand_lstm(self, lstm_blocks=1, lstm_units=32, dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x,1))
        input_u = keras.Input(shape=(dim_u,1))
        if data_type == '1D':
            inputs = tf.concat([input_x,input_u], axis=1)   # [y1,y2,x1,x2]
            # inputs = tf.transpose(tf.concat([input_x, input_u], axis=1),[0,2,1])
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
        x = keras.layers.Dense(dense_units,activation='relu')(x)
        out = keras.layers.Dense(1,activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out)
        return model

    def stand_gru(self, gru_blocks=1, gru_units=32, dense_units=64, dim_u=1, dim_x=3, data_type='2DT'):
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x, 1))
        input_u = keras.Input(shape=(dim_u, 1))
        if data_type == '1D':
            inputs = tf.transpose(tf.concat([input_x, input_u], axis=1), [0, 2, 1])  # [y1,y2,x1,x2]
            # inputs = tf.concat([input_x, input_u], axis=1)  # [y1,y2,x1,x2]
        elif data_type == '2D':
            if (dim_x + dim_u) % 2 == 0:
                inputs = tf.reshape(tf.concat([input_x, input_u], axis=1),
                                    [-1, int((dim_x + dim_u) / 2), 2])  # [[y1,y2],[x1,x2]]
        elif data_type == '2DT':
            inputs = tf.transpose(tf.reshape(tf.concat([input_x, input_u], axis=1), [-1, int((dim_x + dim_u) / 2), 2]),
                                  [0, 2, 1])  # [[y1,x1],[y2,x2]]

        for j in range(gru_blocks):
            if j == gru_blocks - 1:
                x = keras.layers.GRU(gru_units, return_sequences=False)(inputs)
            else:
                x = keras.layers.GRU(gru_units, return_sequences=True)(inputs)
            inputs = x
        x = keras.layers.Dense(dense_units, activation='relu')(x)
        out = keras.layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out)
        return model

    def stand_bp(self, nblocks=1, units=32, dim_u=1, dim_x=3, data_type='2DT'):
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x, 1))
        input_u = keras.Input(shape=(dim_u, 1))
        if data_type == '1D':
            # inputs = tf.concat([input_x, input_u], axis=1)
            inputs = tf.transpose(tf.concat([input_x, input_u], axis=1),[0,2,1])# [y1,y2,x1,x2]
        elif data_type == '2D':
            if (dim_x + dim_u) % 2 == 0:
                inputs = tf.reshape(tf.concat([input_x, input_u], axis=1),
                                    [-1, int((dim_x + dim_u) / 2), 2])  # [[y1,y2],[x1,x2]]
        elif data_type == '2DT':
            inputs = tf.transpose(tf.reshape(tf.concat([input_x, input_u], axis=1), [-1, int((dim_x + dim_u) / 2), 2]),
                                  [0, 2, 1])  # [[y1,x1],[y2,x2]]
        for j in range(nblocks):
            if j == nblocks - 1:
                x = keras.layers.Dense(int(units/2), activation='relu')(inputs)
            else:
                x = keras.layers.Dense(units, activation='relu')(inputs)
            inputs = x
        out = keras.layers.Dense(1, activation='linear')(x)
        out = out[:,-1]
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out)
        return model

    def stand_linear(self, dim_u=1, dim_x=3):
        # 构建模型,2个输入，分别x和u
        input_x = keras.Input(shape=(dim_x, 1))
        input_u = keras.Input(shape=(dim_u, 1))
        x = tf.concat([input_x, input_u], axis=1)
        # x = tf.transpose(tf.concat([input_x, input_u], axis=1), [0, 2, 1])  # [y1,y2,x1,x2]
        out = keras.layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs=[input_x, input_u], outputs=out)
        return model


if __name__ == '__main__':
    # 示例数据
    u = np.array([i+1 for i in range(100)])  # 系统输出
    y = np.array([i+2 for i in range(100)])  # 系统输出
    # 窗口大小
    input_window_dy = 2
    input_window_du = 2
    dim = input_window_dy + input_window_dy - 1
    # 创建窗口生成器
    window_generator = WindowGenerator(input_window_dy, input_window_du, u, y, u_dim=1)
    # 生成序列
    x_sequences, u_sequences, y_sequences = window_generator.generate_3D_sequences()
    num = int(0.8 * len(x_sequences))
    # 训练模型
    train_x_sequences, train_u_sequences, train_y_sequences = x_sequences[:num,:,:], u_sequences[:num,:,:], y_sequences[:num,:,:]
    test_x_sequences, test_u_sequences, test_y_sequences = x_sequences[num:,:,:], u_sequences[num:,:,:], y_sequences[num:,:,:]
    # 训练模型
    # 构建TCMNet模型
    my_model = NNet(num_blocks= 3, kernel_size= 3, nb_filters = 64)
    model_name = 'NNet'

    # model = my_model.resnet_tcm(lstm_blocks=2, lstm_units=32, dense_units=64, dim_u=1, dim_x=dim, data_type='2DT')
    # model = my_model.skip_tcm(lstm_blocks=2, lstm_units=32, dense_units=64, dim_u=1, dim_x=dim, data_type='2DT')
    # model = my_model.resnet_skip_tcm(lstm_blocks=2, lstm_units=32, dense_units=64, dim_u=1, dim_x=dim, data_type='2DT')
    # model = my_model.resnet_2skip_tcm(lstm_blocks=2, lstm_units=32, dense_units=64, dim_u=1, dim_x=dim, data_type='2DT')
    model = my_model.resnet_tcm_skip(lstm_blocks=2, lstm_units=32, dense_units=64, dim_u=1, dim_x=dim, data_type='2DT')

    # 编译模型
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    # 训练模型
    model.fit([train_x_sequences,train_u_sequences], train_y_sequences, epochs=10, batch_size=128, shuffle=False,
              validation_data=([test_x_sequences, test_u_sequences], test_y_sequences))
    # print(model.predict([x_sequences,u_sequences]))
    # print(model.predict([x_sequences,u_sequences])[:,-1])
    keras.utils.plot_model(model, to_file=f'model_fig/{model_name}.png', show_shapes=True, show_layer_names=True)

    y_pred_train = model.predict([train_x_sequences, train_u_sequences])
    y_pred_test = model.predict([test_x_sequences, test_u_sequences])

    print(model.evaluate([test_x_sequences, test_u_sequences],test_y_sequences))
