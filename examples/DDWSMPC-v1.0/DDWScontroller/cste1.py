import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 模型参数 (与MATLAB示例保持一致)
CSTR_PARAMS = {
    'F': 1.0,  # m³/h
    'V': 1.0,  # m³
    'R': 1.985875,  # kcal/(kmol·K)
    'delta_H': -5960.0,  # kcal/kmol
    'E': 11843.0,  # kcal/kmol
    'k0': 34930800.0,  # 1/h
    'rho_Cp': 500.0,  # kcal/(m³·K)
    'UA': 150.0  # kcal/(K·h)
}


class CSTRModel:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.params = CSTR_PARAMS

    def dynamics(self, state, u):
        """改进的动力学方程，支持批量计算"""
        CA, T = state[..., 0], state[..., 1]
        CAf, Tf, Tc = u[..., 0], u[..., 1], u[..., 2]

        k = self.params['k0'] * tf.exp(-self.params['E'] / (self.params['R'] * T))
        reaction_rate = k * CA

        dCA = (self.params['F'] / self.params['V']) * (CAf - CA) - reaction_rate
        dT = (self.params['F'] / self.params['V']) * (Tf - T)
        dT += (self.params['delta_H'] / self.params['rho_Cp']) * reaction_rate
        dT += (self.params['UA'] / (self.params['V'] * self.params['rho_Cp'])) * (Tc - T)

        return tf.stack([dCA, dT], axis=-1)

    def rk4_step(self, state, u):
        """四阶龙格-库塔积分"""
        k1 = self.dynamics(state, u)
        k2 = self.dynamics(state + 0.5 * self.dt * k1, u)
        k3 = self.dynamics(state + 0.5 * self.dt * k2, u)
        k4 = self.dynamics(state + self.dt * k3, u)
        return state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, initial_state, u_sequence):
        """开环模拟"""
        states = [initial_state.numpy()] if tf.is_tensor(initial_state) else [initial_state]
        state = tf.convert_to_tensor(initial_state, dtype=tf.float32)

        for u in u_sequence:
            u_tensor = tf.convert_to_tensor(u, dtype=tf.float32)
            state = self.rk4_step(state, u_tensor)
            states.append(state.numpy())
        return np.array(states)


class MPCController:
    def __init__(self, model, horizon=20):
        self.model = model
        self.horizon = horizon
        self.Tc_min = 280.0
        self.Tc_max = 350.0

        # 优化器配置
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.05,
            clipnorm=1.0
        )

    @tf.function
    def _compute_trajectory(self, initial_state, Tc_seq):
        """计算预测轨迹（图模式加速）"""
        states = tf.TensorArray(tf.float32, size=self.horizon + 1)
        states = states.write(0, initial_state)

        for t in range(self.horizon):
            u = tf.stack([10.0, 300.0, Tc_seq[t]], axis=0)
            next_state = self.model.rk4_step(states.read(t), u)
            states = states.write(t + 1, next_state)
        return states.stack()

    def mpc_step(self, current_state, setpoint, max_iter=200):
        """带约束的MPC优化"""
        # 初始化控制序列（使用当前状态线性插值）
        current_state = tf.convert_to_tensor(current_state, dtype=tf.float32)
        Tc_init = tf.linspace(current_state[1], setpoint, self.horizon)
        Tc_var = tf.Variable(Tc_init, trainable=True, dtype=tf.float32)

        # 定义优化问题
        @tf.function
        def cost_fn():
            # 应用输入约束
            Tc_clipped = tf.clip_by_value(Tc_var, self.Tc_min, self.Tc_max)
            pred_states = self._compute_trajectory(current_state, Tc_clipped)

            # 计算代价函数
            tracking_error = tf.reduce_sum(tf.square(pred_states[1:, 1] - setpoint))
            control_penalty = 0.1 * tf.reduce_sum(tf.square(Tc_clipped - 300.0))
            smooth_penalty = 0.05 * tf.reduce_sum(tf.square(Tc_clipped[1:] - Tc_clipped[:-1]))

            return tracking_error + control_penalty + smooth_penalty

        # 执行梯度优化
        for _ in range(max_iter):
            with tf.GradientTape() as tape:
                loss = cost_fn()
            grads = tape.gradient(loss, [Tc_var])
            self.optimizer.apply_gradients(zip(grads, [Tc_var]))

            # 投影法强制约束
            Tc_var.assign(tf.clip_by_value(Tc_var, self.Tc_min, self.Tc_max))

        return float(Tc_var[0].numpy())


# 系统稳态计算（用于初始化）
def find_steady_state(Tc=292.0):
    """计算稳态工作点"""

    def steady_eq(x):
        CA, T = x
        k = CSTR_PARAMS['k0'] * np.exp(-CSTR_PARAMS['E'] / (CSTR_PARAMS['R'] * T))
        dCA = (CSTR_PARAMS['F'] / CSTR_PARAMS['V']) * (10.0 - CA) - k * CA
        dT = (CSTR_PARAMS['F'] / CSTR_PARAMS['V']) * (300.0 - T)
        dT += (CSTR_PARAMS['delta_H'] / CSTR_PARAMS['rho_Cp']) * k * CA
        dT += (CSTR_PARAMS['UA'] / (CSTR_PARAMS['V'] * CSTR_PARAMS['rho_Cp'])) * (Tc - T)
        return np.array([dCA, dT]) ** 2

    res = minimize(lambda x: np.sum(steady_eq(x)), x0=[8.5, 310.0],
                   bounds=[(0, 10), (300, 400)])
    return res.x if res.success else None


# 主程序
if __name__ == "__main__":
    # 初始化模型和控制器
    cstr = CSTRModel(dt=0.01)
    mpc = MPCController(cstr, horizon=15)

    # 计算稳态初始条件
    ss_point = find_steady_state(Tc=292.0)
    print(f"Steady State: CA={ss_point[0]:.4f} kmol/m³, T={ss_point[1]:.4f} K")

    # MPC控制测试
    setpoint = 320.0  # 目标温度
    current_state = tf.constant(ss_point, dtype=tf.float32)
    history = {
        'states': [current_state.numpy()],
        'Tc': [],
        'time': [0.0]
    }

    # 模拟运行1小时（100个控制周期）
    for step in range(100):
        # MPC计算控制量
        Tc_opt = mpc.mpc_step(current_state, setpoint)

        # 应用控制量并模拟系统
        u = np.array([10.0, 300.0, Tc_opt], dtype=np.float32)
        current_state = cstr.rk4_step(current_state, u)
        print(f"{step}: Tc={Tc_opt:.4f},u: {Tc_opt}, CA={current_state[0]:.4f} kmol/m³, T={current_state[1]:.4f}")

        # 记录数据
        history['states'].append(current_state.numpy())
        history['Tc'].append(Tc_opt)
        history['time'].append(history['time'][-1] + cstr.dt)

    # 转换为numpy数组
    history['states'] = np.array(history['states'])
    history['Tc'] = np.array(history['Tc'])
    history['time'] = np.array(history['time'])

    # 可视化结果
    plt.figure(figsize=(12, 9))

    # 温度响应
    plt.subplot(3, 1, 1)
    plt.plot(history['time'], history['states'][:, 1], label='Reactor Temperature')
    plt.plot(history['time'], np.full_like(history['time'], setpoint), 'r--', label='Setpoint')
    plt.ylabel('Temperature (K)')
    plt.legend()

    # 浓度变化
    plt.subplot(3, 1, 2)
    plt.plot(history['time'], history['states'][:, 0], 'g', label='Concentration')
    plt.ylabel('CA (kmol/m³)')
    plt.legend()

    # 控制量变化
    plt.subplot(3, 1, 3)
    plt.step(history['time'][:-1], history['Tc'], 'k', where='post', label='Cooling Temp')
    plt.ylabel('Tc (K)')
    plt.xlabel('Time (h)')
    plt.ylim([280, 350])
    plt.legend()

    plt.tight_layout()
    plt.show()
