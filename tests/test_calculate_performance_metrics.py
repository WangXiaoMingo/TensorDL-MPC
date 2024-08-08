import pandas as pd

if __name__ == '__main__':
    import numpy as np
    from src.dlmpc import calculate_performance_metrics
    data = pd.read_excel('dataset/test_performance.xlsx')
    index = ['BP_mpc']
    time = np.array(data['Time'])+1
    setpoint = np.array(data['reference'])
    sys_out = np.array(data['System_output'])
    result = calculate_performance_metrics(index, sys_out, setpoint, time, percent_threshold=0.02,
                                      ise=True,
                                      iae=True,
                                      overshoot=True,
                                      peak_time=True,
                                      rise_time=True,
                                      rise_time1=True,
                                      settling_time=True,
                                      steady_state_error=True)
    print(result)

