# dlmpc/utils/regression_metrics.py

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


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,explained_variance_score
import pandas as pd
import numpy as np

pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

def Calculate_Regression_metrics(true_value, predict, label='train'):
    """
    Calculate regression metrics for comparing true and predicted values.

    Parameters:
    true_value -- Array of true values
    predict -- Array of predicted values
    label -- String label for the result set (default is 'train')

    Returns:
    result -- Pandas DataFrame containing the calculated metrics
    """
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(true_value, predict)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(true_value, predict)

    # Calculate R-squared (R2) Score
    r2 = r2_score(true_value, predict)

    # Calculate Explained Variance Score
    ex_var = explained_variance_score(true_value, predict)

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(true_value, predict)

    # Create a DataFrame to hold the results
    result = pd.DataFrame({
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'ex_var': ex_var,
        'mape': mape
    }, index=[label])
    return result


