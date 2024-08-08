import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use('TkAgg')
import pandas as pd

mpl.rcParams['lines.linewidth'] = 1

# 设置新罗马字体
fontsize = 18
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': fontsize, }

# python与simulink交互，python充当控制器，simulink充当物理对象，python将计算好的控制器输出反馈给simulink，

def fig_plot(data,fig_columns,figsize=(14, 16)):
    import math
    algorithms = np.array(data.columns)
    subfig = [chr(ord('a') + i) for i in range(data.shape[1])]

    default_blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    fig, axs = plt.subplots(nrows=math.ceil(data.shape[1]/fig_columns),ncols=fig_columns,figsize=figsize)


    for index, algorithm in enumerate(algorithms):

        plt.subplots_adjust(hspace=0.3,wspace=0.25)
        # 绘制主图
        algorithm_name = f'{algorithm}'
        data_temp = data[f'{algorithm}']
        axs[index].plot(np.array(data_temp), label=f'{algorithm_name}')


        axs[index].tick_params(direction='in')
        # axs[index // 2][index % 2].set_ylim([-25, 25])
        axs[index].set_xlim([0, len(data.values)])
        # axs[index].set_xlim([0, 100000])
        # axs[index].set_ylim([-1, max(data_temp)*1.1])

        # axs[index].set_title('(' + subfig[index] + ')',color=default_blue, loc='left', ha='left', va='baseline',
        #                                      fontdict=font)
        axs[index].set_xlabel('Sample', fontdict=font)
        axs[index].xaxis.set_label_coords(0.5, -0.15)  # 调整x轴标签的位置
        axs[index].set_ylabel(algorithm_name, fontdict=font)
        # axs[index].legend(prop=font, ncol=fig_columns, loc='upper right')
        labels = axs[index].get_xticklabels() + axs[index].get_yticklabels()
        for label in labels:
            label.set_fontname('Times New Roman')
            label.set_fontsize(fontsize)
        # print(np.array(data.index))
        if index == 0:
            axs[index].set_ylim([-1, max(data_temp)*1.1])
        # axs[index].set_yticks([i*500 for i in range(int(max(data_temp)*1.1)) if i == i%500])
        # axs[index].set_yticklabels(['0', '30000', '40000', '50000', '60000','70000','80000','90000','100000'], fontdict=font)

    plt.tick_params(labelsize=fontsize)
    plt.tight_layout()
    plt.savefig('../fig/{}.pdf'.format('KTTResult'), dpi=500, bbox_inches='tight')  # 保存图片
    plt.show()

if __name__ == '__main__':
    data = pd.read_excel('../data/KTT_7.xlsx', header=0, index_col='时间')
    # data = data[['下料量','转速','进风管压力','窑尾']].iloc[:2000]
    data = data[['下料量', '窑尾']]
    print(data)

    data.columns = ['Blanking Volume','KTT(℃)']
    # data = data.reset_index()

    print(data.describe())
    fig_plot(data, 1, figsize=(8, 5))
