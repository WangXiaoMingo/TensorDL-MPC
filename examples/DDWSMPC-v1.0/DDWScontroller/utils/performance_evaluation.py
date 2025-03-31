# dlmpc/utils/performance evaluation.py

# Software Copyright Notice

#   This file is part of DL-MPC to calculate the system control performance evaluation
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

import numpy as np
import pandas as pd

# Calculate the Integral of the Squared Error (ISE)
# ISE = mean(sum((ref - sys)**2))
def calculate_ise(sys, ref):
    ise = np.mean((ref - sys)**2)
    return ise

# Calculate the Integral of the Absolute Error (IAE)
# IAE = mean(sum(abs(ref - sys)))
def calculate_iae(sys, ref):
    iae = np.mean(np.abs(ref - sys))
    return iae

# Calculate the Overshoot
# Overshoot = (peak value - setpoint) / setpoint * 100%
def calculate_overshoot(sys_out, setpoint):
    # Find the maximum value in the system output array
    max_value = np.max(sys_out)
    overshoot = (max_value - setpoint[-1]) / setpoint[-1] * 100
    return str(np.round(overshoot, 2)) + ' %'

# Get the Peak Time
# Peak Time = the time at which the system response curve reaches its peak value
def calculate_peak_time(time, sys_out):
    # Find the index of the maximum value in the system output array
    max_index = np.argmax(sys_out)
    # Find the time corresponding to the maximum value index
    peak_time = time[max_index]
    return peak_time

# Rise Time
# Rise Time = the time required for the system response to go from 10% to 90% of the final value
def calculate_rise_time(time, sys_out, setpoint):
    # Find the indices where the system output crosses 10% and 90% of the setpoint
    ten_percent = 0.1 * setpoint[-1]
    ninety_percent = 0.9 * setpoint[-1]
    ten_percent_index = np.where(sys_out >= ten_percent)[0][0]
    ninety_percent_index = np.where(sys_out >= ninety_percent)[0][0]
    # Calculate the rise time as the difference between these time indices
    rise_time = time[ninety_percent_index] - time[ten_percent_index]
    return rise_time

# Rise Time
# Rise Time: The time it takes for the system response curve to reach the 90% of the final value from its initial value.
def calculate_rise_time1(time, sys_out, setpoint):
    # Find the index of the system output array where it first exceeds the 90% of the final value
    ninety_percent = 0.9 * setpoint[-1]
    cross_index = np.where(sys_out >= ninety_percent)[0][0]
    # Get the rise time
    rise_time = time[cross_index]
    return rise_time

# Settling Time
# Settling Time: The time it takes for the system response curve to fluctuate near the steady-state value.
def calculate_settling_time(time, sys_out, setpoint, percent_threshold=0.02):
    # Calculate the fluctuation range threshold
    threshold = sys_out[-1] * percent_threshold
    # Get the settling time
    try:
        # Find the index where the system output first enters the target value range
        cross_index = np.where(np.abs(sys_out - setpoint) <= threshold)  # All indices less than the threshold
        cross_index_1 = np.where(cross_index[0][1:] - cross_index[0][:-1] != 1)  # Find indices that are not equal to 1
        last_index = cross_index[0][cross_index_1[-1] + 1][-1]  # The index where the system enters the steady state
        settling_time = time[last_index]
    except:
        cross_index = np.where(np.abs(sys_out - setpoint) <= threshold)[0][0]  # All indices less than the threshold
        settling_time = time[cross_index]
    return settling_time


# Steady-State Error
# Steady-State Error: The steady-state error is the deviation between the system's output and the target value after the system has reached a stable state.
def calculate_steady_state_error(sys_out, setpoint):
    # The steady-state error is the difference between the last value of the system output sequence and the target value.
    steady_state_error = abs(sys_out[-1] - setpoint[-1])
    return steady_state_error


def calculate_performance_metrics(index,sys_out, setpoint,time,percent_threshold=0.02,
                                  ise=True,
                                  iae=True,
                                  overshoot=True,
                                  peak_time=True,
                                  rise_time=True,
                                  rise_time1=True,
                                  settling_time=True,
                                  steady_state_error=True):
    result = pd.DataFrame()
    result.index = index
    if ise:
        result['ISE'] = calculate_ise(sys_out, setpoint)
    if iae:
        result['IAE'] = calculate_iae(sys_out, setpoint)
    if overshoot:
        result['Overshoot'] = calculate_overshoot(sys_out, setpoint)
    if peak_time:
        result['PeakTime'] = calculate_peak_time(time, sys_out)
    if rise_time:
        result['RiseTime'] = calculate_rise_time(time, sys_out, setpoint)
    if rise_time1:
        result['RiseTime1'] = calculate_rise_time1(time, sys_out, setpoint)
    if settling_time:
        result['SettingTime'] = calculate_settling_time(time, sys_out, setpoint, percent_threshold)
    if steady_state_error:
        result['Steady-StateError'] = calculate_steady_state_error(sys_out, setpoint)
    return result



