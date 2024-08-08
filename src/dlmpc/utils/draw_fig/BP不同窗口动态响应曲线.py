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
fontsize = 15
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': fontsize, }

if __name__ == '__main__':
    data = pd.read_excel('../result/BP不同窗口对比.xlsx', index_col='time').iloc[100:,]
    # time	sys_out_0.1	sys_u_0.1	sys_out_0.2	sys_u_0.2	sys_out_0.3	sys_u_0.3	sys_out_0.05	sys_u_0.05
    # sys_out_0.08	sys_u_0.08	sys_out_1	sys_u_1	ref_output
    data['ref_output'] = 620
    data = data.reset_index()
    data_label_y = ['sys_out_1','sys_out_2']
    data_label_u = ['sys_u_1','sys_u_2']
    sub_fig = ['(a)','(b)','(c)']


    fig, axs = plt.subplots(3,1, figsize=(10, 8))  # 子图
    plt.subplots_adjust(hspace=0.2, wspace=0.1)


    for index, value in enumerate(sub_fig):
        if index == 0:      # 画响应曲线
            for sysout in data_label_y:
                axs[index].plot(data[sysout], label=sysout)  #
            # 画一条垂直线
            x = [1, 1]  # 垂直线的 x 坐标范围
            y = [0, max(data['ref_output'])]  # 垂直线的 y 坐标范围
            axs[index].plot(data['ref_output'], 'r', label='ref')  #
            # axs[index].plot(x, y, color='r')  # 使用红色虚线绘制垂直线
            axs[index].set_ylabel('System Output', fontdict=font)


        elif index == 1:    # 画控制输出
            for sysu in data_label_u:
                axs[index].plot(data[sysu], label=sysu)  #
            axs[index].set_ylabel('Control u', fontdict=font)

        elif index == 2: # 画误差曲线图
            for sysout in data_label_y:
                axs[index].plot(data['ref_output'] - data[sysout], label=sysout)  #
            axs[index].set_ylabel('System Error', fontdict=font)
            axs[index].set_xlabel('Time (s)', fontdict=font)

        axs[index].tick_params(direction='in')
        axs[index].set_xlim([0, len(data['ref_output'])])
        labels = axs[index].get_xticklabels() + axs[index].get_yticklabels()
        for label in labels:
            label.set_fontname('Times New Roman')
            label.set_fontsize(fontsize)

    # axs[0].legend(loc='upper center',  bbox_to_anchor=(0.5, 1.15),ncol=6)
    plt.legend(loc='upper center', ncol=3, prop=font)
    plt.tick_params(labelsize=fontsize)
    plt.savefig('../fig/{}.pdf'.format('KTTBPbeta'), dpi=500, bbox_inches='tight')  # 保存图片
    plt.show()