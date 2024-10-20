
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
    import time
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # tf.config.run_functions_eagerly(True)
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
    R = np.eye(control_horizon) * 0.01  # 控制输入的权重矩阵 # 0.01

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
        script_path = 'test_models_bp_regressor.py'
        os.system(f'python {script_path}')

    model = load_model(f'models_save/BP_predictor.h5')
    # model.summary()

    # TODO: Step4: mpc training

    if mpc:
        # 创建MPC控制器实例
        mpc_controller = DeepLearningMPCController(model, predict_horizon, control_horizon, Q, R, ly, lu, dim_u, [-du_bounds,du_bounds],u_bounds, opt)
        data = np.zeros((N - 0, 7))
        result = pd.DataFrame(data,
                              columns=['Time', 'reference', 'System output', 'u', 'solving_time', 'epoch', 'error'])

        # MPC控制循环
        for i in range(0,N):
            if i < 30:
                y_ref = 10
            elif i >= 30 and i < 70:
                y_ref = 5
            elif i >= 70 and i < 100:
                y_ref = 8
            else:
                y_ref = 10

            # 优化控制输入
            t_start = time.time()
            J, u_sequence, epoch = mpc_controller.optimize(error, state_y, state_u, y_ref, iterations = 100, tol = 1e-6)

            solving_time = time.time() - t_start
            # 实施第一个控制输入
            u0 = u_sequence[0]
            # 更新状态
            x_current = np.concatenate((state_y, state_u), axis=0).reshape(1,dim,1)
            y_nn = model.predict([x_current, np.atleast_3d(u0)],verbose = 0)
            # system output
            y_plant = simulation.plant(np.append(tf.squeeze(state_u), u0),tf.squeeze(state_y))
            # 在线校正
            error = mpc_controller.online_correction(y_nn[0][0], y_plant[0])
            print(f'>>> Current Time: {i},\t Object J: {J[0][0]:>=.4f}, \t Current u:{u0[0]:>=.4f}, \t Current NN output:{y_nn[0][0]:>=.4f}, \t Current System output: {y_plant[0]:>=.4f}, \t Optimization epoch: {epoch}, \t Solving time:{solving_time:>=.4f} s')

            # 更新状态
            state_y = tf.roll(state_y, -1,axis=0)
            state_u = tf.roll(state_u, -1,axis=0)
            state_y = tf.tensor_scatter_nd_update(state_y, [[ly-1, 0]], y_plant)
            state_u = tf.tensor_scatter_nd_update(state_u, [[lu-1, 0]], u0)

            result.at[i, 'Time'] = i
            result.at[i, 'reference'] = y_ref
            result.at[i, 'System output'] = y_plant
            result.at[i, 'u'] = u0
            result.at[i, 'epoch'] = epoch
            result.at[i, 'solving_time'] = solving_time
            result.at[i, 'error'] = y_ref - y_plant

        plt.figure()
        plt.plot(result['reference'], '-', label='reference')  # 绘制当前变量的数据
        plt.plot(result['System output'], '--', label='System output')  # 绘制当前变量的数据
        plt.legend(loc='upper right')
        plt.show()


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





