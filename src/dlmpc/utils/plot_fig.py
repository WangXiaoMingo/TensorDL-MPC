# dlmpc/utils/plot_fig.py

# Software Copyright Notice

#   This file is part of DL-MPC
#
#   DL-MPC: A toolbox for deep learning-based nonlinear model predictive control
#
#   Copyright (c) 2024, Xiaoming Wang. All rights reserved
#
#   This software (including but not limited to all modules, files, and code) is developed and owned by Xiaoming Wang.
#   Unauthorized distribution, copying, modification, or redistribution of this software in any form is prohibited.
#
#   This software is freely available for academic research activities, including but not limited to paper writing,
#   academic discussions, and academic conferences.
#
#   Please note that this software is not intended for commercial use, including but not limited to commercial projects,
#   products, and services. If you require the use of this software for commercial purposes, please contact Xiaoming Wang
#   to obtain the appropriate license.
#
#   This software may contain third-party software components, the use and distribution of which are subject to the
#   respective third-party license agreements. If you need to use third-party software components,
#   please ensure compliance with their license agreements.
#
#   If you have any questions about the software or require further assistance,
#   please contact Xiaoming Wang support team. e-mail: wangxiaoming19951@163.com
#
#   Last updated on June, 2024
#   Author: Xiaoming Wang

import matplotlib.pyplot as plt
import matplotlib as mpl

# Define font styles
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
font2 = {'family': 'STSong', 'weight': 'normal', 'size': 13}
fontsize1 = 13

# Set fonts to support Chinese characters in the plot
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def plot_line(predict, true_value, figure_property):
    """
    Plot a line chart to compare predicted values with true values.

    Parameters:
    predict -- List or array of predicted values
    true_value -- List or array of true values
    figure_property -- Dictionary containing chart title and axis labels

    Note: Ensure that predict, true_value, and figure_property variables are defined before using the function.
    eg: figure_property = {'title': model_name, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}


    """
    # Create figure and axis objects
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the predicted and true values
    ax.plot(predict, label='Predicted Value')
    ax.plot(true_value, label='True Value')

    # Set the chart title and axis labels
    ax.set_title(figure_property['title'], fontdict=font2)
    ax.set_xlabel(figure_property['X_label'], fontdict=font2)
    ax.set_ylabel(figure_property['Y_label'], fontdict=font2)

    # Set the font size for tick labels
    plt.tick_params(labelsize=12)

    # Set the font for tick labels to Times New Roman
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')

    # Add a legend
    plt.legend(prop=font2)

    # Adjust the layout
    plt.tight_layout()

    # Turn off the grid
    ax.grid(False)

    # Optional: Save the figure
    # plt.savefig('../fig/{}.jpg'.format(figure_property['title']), dpi=500, bbox_inches='tight')

    # Show the plot
    plt.show()
