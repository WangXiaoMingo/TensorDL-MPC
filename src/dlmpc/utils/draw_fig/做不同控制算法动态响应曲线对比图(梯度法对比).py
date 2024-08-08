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
fontsize = 13
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': fontsize, }

if __name__ == '__main__':
    '''1.======================绘制不同算法曲线=========================='''

    # data = pd.read_excel('../result/KTTControl_robuster.xlsx', index_col='time').iloc[150:501,:]
    data = pd.read_excel('../result/CloseloopRobust.xlsx', index_col='time').iloc[0:300, :]
    data['ref_output'] = 620
    data = data.reset_index()

    data_label = {}
    algorithms  = ['NMPC','Linear','BP_MPC','GRU_MPC','LSTM_MPC','TCM-MPC']
    data_label['data_label_bp']   = {'NMPC': 'mpc_y','Linear-MPC': 'Linear_y',  'BP-MPC':   'BP_y',   'BP-AMPC':   'ABP_y'}        # 'Linear_AMPC': 'ALinear_y',
    data_label['data_label_GRU']  = {'NMPC': 'mpc_y', 'Linear-MPC': 'Linear_y', 'GRU-MPC':  'GRU_y',  'GRU-AMPC':  'AGRU_y'}
    data_label['data_label_LSTM'] = {'NMPC': 'mpc_y', 'Linear-MPC': 'Linear_y', 'LSTM-MPC': 'LSTM_y', 'LSTM-AMPC': 'ALSTM_y'}
    data_label['data_label_DNN']  = {'NMPC': 'mpc_y', 'Linear-MPC': 'Linear_y', 'TCM-MPC':  'TCM_y', 'TCM-AMPC': 'ATCM_y'}

    line_style = ['--','-.', ':','-', '-'] # 使用自定义虚线样式], dashes=(5, 2)

    sub_fig = ['(a)','(b)','(c)','(d)']

    fig, axs = plt.subplots(2,2, figsize=(10, 7))  # 子图
    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    for index, value in enumerate(data_label):
        for style,(label, sysout) in enumerate(data_label[value].items()):
            axs[index//2][index%2].plot(data[sysout],label=label)  # ,linestyle=line_style[style]
            # axs[index // 2][index % 2].plot(data[sysout],linestyle=line_style[style], label=label)  #
        # 画一条垂直线
        axs[index//2][index%2].plot([0, 50], [600, 600], 'k--', label='reference')  # 使用红色虚线绘制垂直线
        x = [50, 50]  # 垂直线的 x 坐标范围
        y = [600, max(data['ref_output'])]  # 垂直线的 y 坐标范围
        axs[index//2][index%2].plot([50, len(data['ref_output'])], [620, 620], 'k--')  #, label='reference'
        axs[index//2][index%2].plot(x, y, 'k--')  # 使用红色虚线绘制垂直线
        # axs[index//2][index%2].plot(x, y, color='k')  # 使用红色虚线绘制垂直线
        # axs[index // 2][index % 2].axhline(y = 15.3,xmin=0, color='k')  # label='reference'
        axs[index//2][index%2].set_ylabel('KTT (℃)', fontdict=font)
        axs[index // 2][index % 2].set_xlabel('Time (minute)', fontdict=font)
        axs[index//2][index%2].legend(ncol=1, prop=font)  #bbox_to_anchor=(0.5, 1.45)\
        axs[index // 2][index % 2].set_title(sub_fig[index], color='black', loc='left', va='baseline',
                         fontdict=font)



        axs[index//2][index%2].tick_params(direction='out')
        axs[index//2][index%2].set_xlim([0, len(data['ref_output'])])
        axs[index//2][index%2].set_xticks([0, 50, 100, 150, 200, 250, 300], fontdict=font)
        # axs[index // 2][index % 2].set_ylim([-1, 20])
        labels = axs[index//2][index%2].get_xticklabels() + axs[index//2][index%2].get_yticklabels()
        for label in labels:
            label.set_fontname('Times New Roman')
            label.set_fontsize(fontsize)

    plt.tick_params(labelsize=fontsize)
    plt.savefig('../fig/{}.pdf'.format('KTTcloseloopCurve'), dpi=800, bbox_inches='tight')  # 保存图片
    plt.show()