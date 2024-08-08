import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
print(pd.__version__)


# 参数设置
# mpl.use('TkAgg')
mpl.rcParams['lines.linewidth'] = 1.2
# mpl.rcParams["font.weight"]='bold'
from matplotlib.patches import Ellipse

# 设置新罗马字体
fontsize = 16
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': fontsize, }

# 实际值	BP预测值	Linear预测值	Linear预测值1	GRU预测值	LSTM预测值
data = pd.read_excel('../result/算法预测结果对比.xlsx')
algorithms = ['Linear','BP','GRU', 'LSTM','TCM']

Linear_error = data['实际值'] - data['Linear预测值']
BP_error =  data['实际值'] - data['BP预测值']
GRU_error =  data['实际值'] - data['GRU预测值']
LSTM_error =  data['实际值'] - data['LSTM预测值']
TCM_error = data['实际值'] - data['TCM预测值']

data = pd.concat([Linear_error,BP_error,GRU_error,LSTM_error,TCM_error],axis=1)
data.columns = algorithms
subfig = ['(a)','(b)']
print(data)

# # 获取四分位数，返回前25%，前50%，前75%和最大值
# q1 = data1.quantile(0.25)
# q3 = data1.quantile(0.75)
# IQR = (q3-q1)
# Maxium = q3 + 1.5 * IQR
# Minium = q1 - 1.5 * IQR
# data1 = data1[(data1 >= Minium) & (data1 <= Maxium)]

import seaborn as sns
default_blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
def fig_box_plot():
    fig, axs = plt.subplots(1,2,figsize=(10, 4))    # 创建箱线图
    plt.subplots_adjust(hspace=1, wspace=3)
    bp = axs[0].boxplot(data, patch_artist=True,showfliers=False) #
    axs[0].axhline(0, color ='k',linestyle='-.', linewidth=0.5)
    axs[0].tick_params(direction='in')
    # axs[1].set_xlim([1, 5])

    # 设置 x 轴刻度标签
    axs[0].set_xticklabels(algorithms)
    # plt.xticks(rotation=15)


    # 设置中位数、上四分位数和下四分位数的标记样式和颜色
    for marker in bp['medians']:
        marker.set_marker('o')
        marker.set_color('#ff5252')

    for marker in bp['whiskers']:
        marker.set_marker('')
        marker.set_linestyle('-')
        marker.set_color('#000000')

    # 设置箱体的填充颜色和边框颜色
    colors = ['#b3e5fc', '#64b5f6', '#2196f3']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('#000000')

    axs[0].tick_params(direction='in')
    axs[0].set_title(subfig[0] ,color='black', loc='left', va='baseline',
                                         fontdict=font)
    axs[0].set_xticks([1,2,3,4,5])
    axs[0].set_xticklabels(['Linear','BP','GRU', 'LSTM','TCM'], fontdict=font)
    axs[0].set_ylabel('Predict Error', fontdict=font)
    axs[0].set_xlabel('Models', fontdict=font)


    # 绘制正态分布曲线
    mu = np.mean(data)
    sigma = np.std(data)  # 均值和标准差

    for model in algorithms:
        x = np.linspace(mu[model] - 5 * sigma[model], mu[model] + 5 * sigma[model], len(data))
        y = 1 / (np.sqrt(2 * np.pi) * sigma[model]) * np.exp(-(x - mu[model]) ** 2 / (2 * sigma[model] ** 2))
        axs[1].plot(x, y, label=model)
    axs[1].axvline(0, color ='k',linestyle='-.', linewidth=0.5)
    axs[1].tick_params(direction='in')
    axs[1].set_title(subfig[1] ,color='black', loc='left', va='baseline',
                                         fontdict=font)
    # axs[1].set_xlim([-5,5])
    plt.legend(prop=font,ncol=1)

    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels() + axs[1].get_xticklabels() + axs[1].get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(fontsize)

    axs[1].set_ylabel('Probability Density',fontdict=font)
    axs[1].set_xlabel('Normal Distribution of Errors', fontdict=font)
    plt.tight_layout()
    plt.tick_params(labelsize=fontsize)
    plt.savefig('../fig/{}.pdf'.format('ProbabilityDensityKTT'), dpi=500, bbox_inches='tight')  # 保存图片
    plt.show()


if __name__ == '__main__':
    fig_box_plot()


