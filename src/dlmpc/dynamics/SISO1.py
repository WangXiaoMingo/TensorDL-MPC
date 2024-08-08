import numpy as np
import matplotlib.pyplot as plt

'''

y(k) = 0.2* np.sin (0.5 * (y(k − 1) + y(k − 2)))
        + 0.2 * np.sin (0.5 * (y(k − 2) + y(k − 3))
         + 2 * u(k − 1) + u(k − 2)) + 
         (4 * u(k − 1) + u(k − 2))/(1 + 0.2 * cos (0.2 * (2 * y(k − 1) + y(k − 2)))
         
         −1.01 ≤ y ≤ 2.02, −1 ≤ u ≤ 1.5.

'''

class SystemPlant:
    def __init__(self, initial_state=None,initial_input=None, noise_amplitude=0.1,sine_wave=True):
        self.initial_state = initial_state if initial_state is not None else np.zeros(3)
        self.initial_input = initial_input if initial_input is not None else np.zeros(3)
        self.noise_amplitude = noise_amplitude
        self.sine_wave = sine_wave
        self.y = None
        self.u = None

    def plant(self,u,x):
        '''
        :param u: u=[u(k-3),u(k-2),u(k-1)]=[u0,u1,u2]
        :param x: x= [y(k-3),y(k-2),y(k-1)]=[y0,y1,y2]
        :return:
        '''
        y = 0.2 * np.sin(0.5 * (x[2] + x[1])) + 0.2 * np.sin(0.5 * (x[1] + x[0]) + 2 * u[2] + u[1]) + (4 * u[2] + u[1]) / (1 + 0.2 * np.cos(0.2 * (2 * x[2] + x[1])))+ np.random.rand(1) * self.noise_amplitude
        return y

    def generate_data(self, n_samples):
        # 初始化系统状态
        self.y = np.zeros(n_samples)
        self.u = np.zeros(n_samples)
        e = self.noise_amplitude * np.random.randn(n_samples)
        # 初始化系统状态和初始化控制输入
        self.y[:3] = self.initial_state
        self.u[:3] = self.initial_input

        if self.sine_wave:
            self.u = np.sin(np.linspace(0, 4 * np.pi, n_samples)) + e
        else:
            for k in range(3, n_samples):
                # 计算控制输入
                self.u[k] = 0.5 * self.u[k - 1] + 0.2 * self.u[k - 2] + e[k - 1]

        # 产生数据
        for k in range(3, n_samples):
            # 计算系统输出
            self.y[k] = self.plant(self.u[k-3:k],self.y[k-3:k])
        return self.y, self.u

    def plot_results(self):
        if self.y is None or self.u is None:
            raise ValueError("Simulation data has not been generated.")

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.y, label='System Output')
        plt.legend()
        plt.title('System Output')

        plt.subplot(2, 1, 2)
        plt.plot(self.u, label='Control Input')
        plt.legend()
        plt.title('Control Input')

        plt.tight_layout()
        plt.show()


# 使用类
if __name__ == '__main__':
    # 初始化系统状态
    import pandas  as pd
    initial_state = np.array([1, 1.2, 0.8])    # 初始状态可以自定义，y[0]，y[1]
    initial_input = np.array([0,0.1, 0.2])  # 初始状态可以自定义，u[0],u[1]
    simulation = SystemPlant(initial_state, initial_input, noise_amplitude=1,sine_wave=False)  # 信号生成方式
    y, u = simulation.generate_data(1000)
    simulation.plot_results()
    data = pd.DataFrame(u,index=[f'k_{i}' for i in range(len(u))], columns= ['u'])
    data['y'] = y
    print(data)



