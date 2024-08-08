import numpy as np
import matplotlib.pyplot as plt

class SystemPlant:
    def __init__(self, initial_state=None,initial_input=None, noise_amplitude=0.1,sine_wave=True):
        self.initial_state = initial_state if initial_state is not None else np.zeros(2)
        self.initial_input = initial_input if initial_input is not None else np.zeros(2)
        self.noise_amplitude = noise_amplitude
        self.sine_wave = sine_wave
        self.y = None
        self.u = None

    def plant(self,u,x):
        '''u0: u[k-9], u[1]:u[k-7],x[0]:y[k-2],x[1]:y[k-1]'''
        y = 0.828 * x[1] - 0.5 * x[0] + 1.20 * u[1] + 1.52 * u[0]
        # y[k] = 0.828 * y[k - 1] - 0.5 * y[k - 2] + 1.20 * u[k - 7] + 1.52 * u[k - 9]
        return y

    def generate_data(self, n_samples):
        # 初始化系统状态
        self.y = np.zeros(n_samples)
        self.u = np.zeros(n_samples)
        # self.u = np.ones(n_samples)*10
        self.u[0] = 1.2
        self.u[1] = 1.5
        e = self.noise_amplitude * np.random.randn(n_samples)
        for k in range(2, n_samples):
            # 计算控制输入
            self.u[k] = 0.5 * self.u[k - 1] + 0.2 * self.u[k - 2] + e[k - 1]

        # 产生数据
        for k in range(9, n_samples):
            u = (self.u[k - 9],self.u[k - 7])
            x = (self.y[k - 2],self.y[k - 1])
            # 计算系统输出
            self.y[k] = self.plant(u,x)
        return self.y, self.u

    def generate_train_data(self, n_samples):
        # 初始化系统状态
        self.y = np.zeros(n_samples)
        self.u = np.zeros(n_samples)
        self.data = np.zeros((n_samples,5))
        self.u[0] = 1.2
        self.u[1] = 1.5
        e = self.noise_amplitude * np.random.randn(n_samples)
        for k in range(2, n_samples):
            # 计算控制输入
            self.u[k] = 0.5 * self.u[k - 1] + 0.2 * self.u[k - 2] + e[k - 1]

        # 产生数据
        for k in range(9, n_samples):
            u = (self.u[k - 9],self.u[k - 7])
            x = (self.y[k - 2],self.y[k - 1])
            # 计算系统输出
            self.y[k] = self.plant(u,x)
            self.data[k,:] = [self.u[k - 9],self.u[k - 7],self.y[k - 2],self.y[k - 1],self.y[k]]
        return self.data

    def plot_results(self):
        if self.y is None or self.u is None:
            raise ValueError("Simulation data has not been generated.")

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.y, label='System Output')
        plt.legend()
        plt.ylabel('y')
        plt.xlabel('t')
        plt.title('System Output')

        plt.subplot(2, 1, 2)
        plt.plot(self.u, label='Control Input')
        plt.legend()
        plt.ylabel('u')
        plt.xlabel('t')
        plt.title('Control Input')
        plt.tight_layout()
        plt.show()


# 使用类
if __name__ == '__main__':
    import pandas  as pd
    # 初始化系统状态
    # initial_state = np.array([1, 1.2])    # 初始状态可以自定义，y[0]，y[1]
    # initial_input = np.array([0.1, 0.2])  # 初始状态可以自定义，u[0],u[1]
    simulation = SystemPlant(noise_amplitude=1)  # 信号生成方式
    y, u = simulation.generate_data(1000)
    data = simulation.generate_train_data(1000)
    print(data)
    simulation.plot_results()
    data = pd.DataFrame(u,index=[f'k_{i}' for i in range(len(u))], columns= ['u'])
    data['y'] = y
    print(data)
