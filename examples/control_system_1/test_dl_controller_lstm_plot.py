
''' test bp-mpc'''
import pandas as pd



if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    from src.dlmpc import SimSISO
    from src.dlmpc import DeepLearningMPCController
    from src.dlmpc import calculate_performance_metrics
    from src.dlmpc import optimizer
    from src.dlmpc import OnlineOptimize
    import time
    import os
    import matplotlib as mpl
    mpl.use('TkAgg')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    plt.tick_params(labelsize=12)

    # TODO: Step1: parameters Settings
    #  NN parametes Settings
    # 窗口大小
    input_window_dy = 2
    input_window_du = 2
    ly = input_window_dy  # y的历史状态的长度
    lu = input_window_du-1  # u的历史输入的长度
    dim = ly+lu

    # MPC parameters Settings
    # Train_NN = True
    Train_NN = False
    mpc = True
    predict_horizon = 4 #10  # 预测时域  (2,1), (4,2 *),(3,3)
    control_horizon = 2 #5   # control_horizon
    dim_u = 1             # control variable dimension = 1
    du_bounds = 10       # control variable constraints (delta u)
    u_bounds = [-5,5]          # control variable constraints (u)
    opt = optimizer(optimizer_name='sgd', learning_rate=0.1, du_bound=None, exponential_decay=False)
    error = 0

    # 定义权重矩阵
    Q = np.eye(predict_horizon) * 0.1   # 跟踪误差的权重矩阵  # 0.1
    R = np.eye(control_horizon) * 0  # 控制输入的权重矩阵 # 0.01

    N = 150  # 运行周期
    y_ref = 10  # 参考轨迹值
    # 初始化系统状态
    '''
    initial_state = np.array([1, 1.2])    # 初始状态可以自定义，y[0]，y[1]
    initial_input = np.array([0.1, 0.2])  # 初始状态可以自定义，u[0],u[1]
    '''
    state_y = tf.constant([[1], [1.2]], dtype=tf.float32)
    state_u = tf.constant([[0.1]], dtype=tf.float32)
    u0 = tf.constant([0.2], dtype=tf.float32)


    # TODO: Step2: load plant and parameters
    '''return plant: simulation'''
    simulation = SimSISO(noise_amplitude=0)

    # TODO: Step3: Load  NN model and training
    if Train_NN:
        '''get trained model'''
        import os
        script_path = 'test_models_lstm_regressor.py'
        os.system(f'python {script_path}')

    model = load_model(f'models_save/LSTM_predictor.h5')
    # model.summary()

    # TODO: Step4: mpc training

    if mpc:
        # 创建MPC控制器实例
        mpc_controller = DeepLearningMPCController(model, predict_horizon, control_horizon, Q, R, ly, lu, dim_u, [-du_bounds,du_bounds],u_bounds, opt)
        data = np.zeros((N-2,7))
        result = pd.DataFrame(data,columns=['Time','reference', 'System output','u','solving_time', 'epoch','error'])
        # 初始化图表
        plt.close()
        fig, ax = plt.subplots(3, 1)
        plt.ion()  # 打开交互模式
        # MPC控制循环
        for i in range(2,N):
            if i > 30:
                y_ref = 5  # 参考轨迹值
            if i > 70:
                y_ref = 10  # 参考轨迹值
            mpc = OnlineOptimize(mpc_controller,state_y, state_u, use_online_correction=True)
            # controller computation
            parameter = mpc.make_step(error, y_ref, iterations=100, tol=1e-6)  #
            u0 = parameter['u0']
            # system output
            plant_output = simulation.plant(np.append(tf.squeeze(state_u), parameter['u0']),tf.squeeze(state_y))
            # estimate state
            state_y, state_u, error = mpc.estimate(parameter['u0'], plant_output)


            print(f">>> Current Time: {i},\t Object J: {parameter['solving_J']:>=.4f}, \t Current u:{parameter['u0'][0]:>=.4f}, \t Current System output: {plant_output[0]:>=.4f}, \t Optimization epoch: {parameter['solving_epoch']}, \t Solving time:{parameter['solving_time']:>=.4f} s")

            result.at[i, 'Time'] = i
            result.at[i, 'reference'] = y_ref
            result.at[i, 'System output'] = plant_output
            result.at[i, 'u'] = parameter['u0']
            result.at[i, 'epoch'] = parameter['solving_epoch']
            result.at[i, 'solving_time'] = parameter['solving_time']
            result.at[i, 'error'] = y_ref - plant_output

            # 动态更新图表
            for a in ax:
                a.clear()
            ax[0].plot(result['reference'][:i],'-',label='reference')    # 绘制当前变量的数据
            ax[0].plot(result['System output'][:i],'--',label='System output')  # 绘制当前变量的数据
            ax[0].legend(loc='upper right')

            ax[1].plot(result['u'][:i],'--',label='u') #
            ax[1].legend(loc='upper right')

            ax[2].plot(result['error'][:i],'--',label='error')
            ax[2].legend(loc='upper right')

            ax[0].set_ylabel('y') #fontdict= font2
            ax[0].set_xlabel('Time')

            ax[1].set_ylabel('u') #fontdict= font2
            ax[1].set_xlabel('Time')

            ax[2].set_ylabel('error') #fontdict= font2
            ax[2].set_xlabel('Time')
            # Set the font for tick labels to Times New Roman
            labels = ax[0].get_xticklabels() + ax[1].get_xticklabels() + ax[2].get_xticklabels() + ax[0].get_yticklabels() + ax[1].get_yticklabels() + ax[2].get_yticklabels()
            for label in labels:
                label.set_fontname('Times New Roman')
            # Adjust the layout
            plt.tight_layout()
            plt.draw()  # 绘制更新
            plt.pause(0.01)  # 暂停短时间，等待更新

        # 控制循环结束后，关闭图表
        plt.ioff()
        plt.close(fig)

        # plt.figure()
        plt.plot(result['reference'], '-', label='reference')  # 绘制当前变量的数据
        plt.plot(result['System output'], '--', label='System output')  # 绘制当前变量的数据
        plt.legend(loc='upper right')
        plt.show(block=False)

        plt.close()


        time = np.array(result['Time'])
        setpoint = np.array(result['reference'])
        sys_out = np.array(result['System output'])

        performance_metrics = calculate_performance_metrics([f'{model.name}_MPC'], sys_out, setpoint, time, percent_threshold=0.02,
                                               ise=True,
                                               iae=True,
                                               overshoot=False,
                                               peak_time=False,
                                               rise_time=False,
                                               rise_time1=False,
                                               settling_time=False,
                                               steady_state_error=False)
        # print(result)

        plt.figure()
        plt.plot(np.array(result['u']), label='input_u')
        plt.show()

        performance_metrics['mean_epoch'] = result['epoch'].mean()
        performance_metrics['mean_solving_time'] = result['solving_time'].mean()
        performance_metrics['mean_error'] = result['error'].mean()
        print(performance_metrics)



