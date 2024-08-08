
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from pylab import *
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,BayesianRidge
from src.dlmpc import Calculate_Regression_metrics
from src.dlmpc import plot_line

# import lightgbm as lgb
from pickle import dump
import warnings
warnings.filterwarnings("ignore")

font1 = {'family': 'Times New Roman','weight': 'normal','size': 13,}
font2 = {'family': 'STSong','weight': 'normal','size': 13,}
fontsize1=13

# 设置字体，以作图显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
# 设置显示属性
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',100)
pd.set_option('display.width',1000)           #宽度
np.set_printoptions(suppress=True)
np.set_printoptions(suppress=True)
pd.set_option('precision',4)
np.set_printoptions(precision=4)


'''=============1. 读取数据======================='''
data = pd.read_excel('../data/KTT_7.xlsx', header=0, index_col='时间')
# data = data[['下料量','转速','进风管压力','窑尾']].iloc[:2000]
data = data[['下料量', '窑尾']]

# TODO: 2. generate Window data
input_window_dy = 3
input_window_du = 3

def generate_sequences(data_y,data_u):
    start_seq = max(input_window_dy,input_window_dy)
    num_samples = len(data_y)-start_seq
    state_num = input_window_dy + input_window_dy
    x = np.zeros((num_samples, state_num))  # 系统状态
    y = np.zeros((num_samples, 1))
    for i in range(start_seq, len(data_y)):
        # 生成状态数据窗口,self.x由两部分构成
        temp_y_state = data_y[i - input_window_dy: i]
        temp_u_state = data_u[i - input_window_du: i]
        temp_state = np.concatenate((temp_y_state, temp_u_state), axis=0)  # 拼接得到状态x
        x[i - start_seq] = temp_state
        # 生成输出窗口
        y[i - start_seq] = data_y[i]
    return x,y

data_u, data_y = np.array(data['下料量']), np.array(data['窑尾'])
x,y = generate_sequences(data_y,data_u)

train_numbers = int(0.8 * len(x))

X_train = x[:train_numbers,:]
Y_train = y[:train_numbers,:]

X_validation = x[train_numbers:,:]
Y_validation = y[train_numbers:,:]
print('数据形状：X_train:{}，X_validation:{},Y_train:{},Y_validation:{}'.format(X_train.shape, X_validation.shape,
                                                                              Y_train.shape, Y_validation.shape))

'''===============3.导入模型并训练=============='''
pipelines = {}
pipelines['EN'] = Pipeline([('scar', StandardScaler()), ('EN', ElasticNet())])

import time
for algorithm in pipelines:
    print(f'\n{algorithm}算法')
    a = time.time()
    clf = pipelines[algorithm].fit(X_train, Y_train)
    print(f'\n{algorithm}算法运行时间{time.time() - a}')

    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_validation)

    predict = np.append(train_predict,test_predict)
    y_true = np.append(Y_train, Y_validation)

    # 保存模型
    model_file = f'../models_save/{algorithm}.sav'
    with open(model_file, 'wb') as model_f:
        dump(clf, model_f)

    '''================计算模型计算训练集结果=============='''
    train_result = Calculate_Regression_metrics(Y_train, train_predict, label='训练集')
    title = '{}算法训练集结果对比'.format(algorithm)
    figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    # performance.figure_plot(train_predict, Y_train, figure_property)

    '''================计算模型计算测试集结果=============='''
    test_result =  Calculate_Regression_metrics(Y_validation, test_predict, label='测试集')
    title = '{}算法测试集结果对比'.format(algorithm)
    figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    # performance.figure_plot(test_predict, Y_validation, figure_property)

    '''===================保存计算结果================'''
    result = pd.concat([train_result, test_result], axis=0)
    print('{}算法计算结果'.format(algorithm))
    print(result)
    result.to_excel('../model_result/{}算法计算结果.xlsx'.format(algorithm))

    title = '{}算法结果对比'.format(algorithm)
    figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    plot_line(predict, y_true, figure_property)


