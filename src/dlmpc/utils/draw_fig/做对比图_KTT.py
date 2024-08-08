import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import os

# mpl.use('TkAgg')
mpl.rcParams['lines.linewidth'] = 1.2
# mpl.rcParams["font.weight"]='bold'
from matplotlib.patches import Ellipse

# 设置新罗马字体
fontsize = 15
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': fontsize, }

# 实际值	BP预测值	Linear预测值	Linear预测值1	GRU预测值	LSTM预测值
data = pd.read_excel('../result/算法预测结果对比.xlsx')
algorithms = ['Linear','BP','GRU', 'LSTM','TCM']

Linear_error = data['实际值'] - data['Linear预测值']
BP_error =  data['实际值'] - data['BP预测值']
GRU_error =  data['实际值'] - data['GRU预测值']
LSTM_error =  data['实际值'] - data['LSTM预测值']
TCM_error = data['实际值'] - data['TCM预测值']

# 绘制主图
fig, ax = plt.subplots(1,1,figsize=(10,4))  # ,figsize=(12,7)
ax.tick_params(direction='in')

# ax.plot(Linear['实际值'], '--',label='True Value')  # ,color='k',color='g'
# ax.plot(Linear['预测值'], '-',  label='Linear')  # ,color='k',color='g'
# ax.plot(BP['预测值'], '-.',  label='BP')  # ,color='k',color='g'
# ax.plot(GRU['预测值'], '--',  label='GRU')  # ,color='k',color='g'
# ax.plot(LSTM['预测值'], '--',  label='LSTM')  # ,color='k',color='g'


ax.plot(Linear_error, '-',  label='Linear')  # ,color='k',color='g'
ax.plot(BP_error, '-.',  label='BP')  # ,color='k',color='g'
ax.plot(GRU_error, ':',  label='GRU')  # ,color='k',color='g'
ax.plot(LSTM_error, '--',  label='LSTM')  # ,color='k',color='g'
ax.plot(TCM_error, '-', dashes=(5, 2),   label='TCM')  # ,color='k',color='g'

ax.set_xlabel('Samples', fontdict=font)
ax.set_ylabel('Value', fontdict=font)


plt.tick_params(labelsize=fontsize)
labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
for label in labels:
    label.set_fontname('Times New Roman')
    label.set_fontsize(fontsize)

# plt.ylim([-0.2, 0.2])
plt.xlim([0, len(data['实际值'].values)])
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': fontsize, }
ax.legend(prop=font2,ncol=5)
plt.savefig('../fig/{}.pdf'.format('KTTerrorResult'), dpi=500, bbox_inches='tight')  # 保存图片
plt.show()