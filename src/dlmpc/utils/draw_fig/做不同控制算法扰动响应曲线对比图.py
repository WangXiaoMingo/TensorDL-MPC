import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import os

# mpl.use('TkAgg')
mpl.rcParams['lines.linewidth'] = 1
# mpl.rcParams["font.weight"]='bold'
from matplotlib.patches import Ellipse

# 设置新罗马字体
fontsize = 18
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': fontsize, }

if __name__ == '__main__':
    '''1.======================绘制不同算法曲线=========================='''

    data = pd.read_excel('../result/CloseloopRobust.xlsx', index_col='time').iloc[0:,:]
    # data = pd.read_excel('../result/KTTControl_robuster.xlsx', index_col='time').iloc[150:400,:]
    data['ref_output'] = 620
    data = data.reset_index()
    data_label_u = []
    data_label_y = []
    algorithms  = ['NMPC','Linear-MPC','BP-MPC','GRU-MPC','LSTM-MPC','TCM-MPC']
    data_label_u = ['mpc_u','Linear_u','BP_u','GRU_u','LSTM_u','TCM_u']
    data_label_y = ['mpc_y','Linear_y','BP_y','GRU_y','LSTM_y','TCM_y']
    sub_fig = ['(a)','(b)','(c)']

    fig, axs = plt.subplots(3,1, figsize=(13, 10))  # 子图
    plt.subplots_adjust(hspace=0.2, wspace=0.1)


    for index, value in enumerate(sub_fig):
        if index == 0:      # 画响应曲线
            for label, sysout in enumerate(data_label_y):
                axs[index].plot(data[sysout], label=algorithms[label])  #
            # 画一条垂直线
            axs[index].plot([0,50], [600,600], 'k--')  # 使用红色虚线绘制垂直线
            x = [50, 50]  # 垂直线的 x 坐标范围
            y = [600, max(data['ref_output'])]  # 垂直线的 y 坐标范围
            axs[index].plot([50,len(data['ref_output'])],[620,620], 'k--', label='reference')  #
            axs[index].plot(x, y, 'k--')  # 使用红色虚线绘制垂直线
            axs[index].set_ylabel('KTT (℃)', fontdict=font)
            axs[index].legend(loc='lower right', ncol=4, prop=font)


        elif index == 1:    # 画控制输出
            for sysu in data_label_u:
                axs[index].plot(data[sysu], label=sysu)  #
            axs[index].set_ylabel('Feed Rate', fontdict=font)
            axs[index].set_yticklabels([10, 13, 16, 19, 22], fontdict=font)

        elif index == 2: # 画误差曲线图
            for sysout in data_label_y:
                axs[index].plot(data['ref_output'] - data[sysout], label=sysout)  #
            axs[index].set_ylabel('KTT Error (℃)', fontdict=font)
            axs[index].set_xlabel('Time (minute)', fontdict=font)

        axs[index].tick_params(direction='in')
        axs[index].set_xlim([0, len(data['ref_output'])])
        labels = axs[index].get_xticklabels() + axs[index].get_yticklabels()
        for label in labels:
            label.set_fontname('Times New Roman')
            label.set_fontsize(fontsize)

    plt.tick_params(labelsize=fontsize)
    plt.savefig('../fig/{}.pdf'.format('KTTcloseloopRobustCurve'), dpi=800, bbox_inches='tight')  # 保存图片
    plt.show()