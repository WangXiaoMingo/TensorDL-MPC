import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 模型参数
F = 1.0  # m³/h
V = 1.0  # m³
R = 1.985875  # kcal/(kmol·K)
delta_H = -5960.0  # kcal/kmol
E = 11843.0  # kcal/kmol
k0 = 34930800.0  # 1/h
rho_Cp = 500.0  # kcal/(m³·K)
UA = 150.0  # kcal/(K·h)


class CSTRModel:
    def __init__(self, dt=0.01):
        self.dt = dt  # 时间步长 (小时)

    def dynamics(self, state, u):
        """CSTR非线性动力学方程"""
        CA, T = state[..., 0], state[..., 1]
        CAf, Tf, Tc = u[..., 0], u[..., 1], u[..., 2]

        reaction_rate = k0 * tf.exp(-E / (R * T)) * CA
        dCA = (F / V) * (CAf - CA) - reaction_rate

        heat_term = (delta_H / rho_Cp) * reaction_rate
        cooling_term = (UA / (V * rho_Cp)) * (Tc - T)
        dT = (F / V) * (Tf - T) + heat_term + cooling_term

        return tf.stack([dCA, dT], axis=-1)

    def generate_data(self, initial_state, u_sequence, steps):
        """生成模拟数据"""
        states = [initial_state]
        state = tf.constant(initial_state, dtype=tf.float32)

        for i in range(steps):
            u = tf.constant(u_sequence[i], dtype=tf.float32)
            derivative = self.dynamics(state, u)
            state = state + self.dt * derivative
            states.append(state.numpy())

        return np.array(states)


class MPCController:
    def __init__(self, model, horizon=10):
        self.model = model
        self.horizon = horizon
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def mpc_step(self, current_state, setpoint):
        """执行MPC优化步骤"""
        # 初始化可训练的控制序列（使用TensorFlow变量）
        Tc_seq = tf.Variable(
            tf.ones(self.horizon, dtype=tf.float32) * 292.0,
            trainable=True
        )
        @tf.function
        def cost_fn():
            state = tf.identity(current_state)  # 创建当前状态的副本
            total_cost = 0.0
            for t in range(self.horizon):
                # 使用TensorFlow操作构建输入（保持所有元素为张量）
                u = tf.stack([
                    tf.constant(10.0, dtype=tf.float32),
                    tf.constant(300.0, dtype=tf.float32),
                    Tc_seq[t]
                ], axis=0)

                # 使用tf.while_loop实现多步预测（更高效）
                state = state + self.model.dt * self.model.dynamics(state, u)

                # 代价函数：跟踪误差 + 控制量变化惩罚
                temp_error = tf.square(state[1] - setpoint)
                control_penalty = 0 * tf.square(Tc_seq[t] - 292.0)
                total_cost += temp_error + control_penalty
            return total_cost


        # 执行优化迭代
        J_prev = tf.constant(np.inf)
        for _ in range(100):
            with tf.GradientTape() as tape:
                loss = cost_fn()
            grads = tape.gradient(loss, [Tc_seq])
            self.optimizer.apply_gradients(zip(grads, [Tc_seq]))
            if tf.abs(loss - J_prev) < 1e-6:
                break


        return Tc_seq.numpy()[0]  # 返回第一个最优控制量


class ImprovedMPCController:
    def __init__(self, model, horizon=10):
        self.model = model
        self.horizon = horizon
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.01,  # 降低学习率
            clipnorm=1.0  # 梯度裁剪
        )
        self.Tc_min = 280.0  # 温度下限
        self.Tc_max = 322.0  # 温度上限

    def rk4_integrate(self, state, u):
        """四阶龙格-库塔积分法"""
        k1 = self.model.dynamics(state, u)
        k2 = self.model.dynamics(state + 0.5 * self.model.dt * k1, u)
        k3 = self.model.dynamics(state + 0.5 * self.model.dt * k2, u)
        k4 = self.model.dynamics(state + self.model.dt * k3, u)
        return state + (self.model.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def mpc_step(self, current_state, setpoint):
        # 初始化控制序列（使用当前温度作为基准）
        Tc_init = tf.ones(self.horizon, dtype=tf.float32) * current_state[1]
        Tc_var = tf.Variable(Tc_init, trainable=True)

        @tf.function  # 启用图执行加速
        def compute_cost(Tc_seq):
            state = tf.identity(current_state)
            total_cost = 0.0
            prev_Tc = Tc_seq[0]

            for t in range(self.horizon):
                # 构建输入向量（保持张量运算）
                u = tf.stack([
                    tf.constant(10.0),
                    tf.constant(300.0),
                    tf.clip_by_value(Tc_seq[t], self.Tc_min, self.Tc_max)
                ], axis=0)

                # 使用RK4进行状态预测
                state = self.rk4_integrate(state, u)

                # 代价函数组成
                tracking_error = tf.square(setpoint-state[1])  # 温度跟踪
                control_magnitude = 0.1 * tf.square(Tc_seq[t] - 292)  # 控制量幅值惩罚
                control_smoothness = 0.05 * tf.square(Tc_seq[t] - prev_Tc)  # 变化率惩罚

                total_cost += tracking_error #+ control_magnitude #+ control_smoothness
                prev_Tc = Tc_seq[t]

            return total_cost

        # 优化循环（增加迭代次数）
        for _ in range(1):
            with tf.GradientTape() as tape:
                current_cost = compute_cost(Tc_var)
            grads = tape.gradient(current_cost, [Tc_var])
            self.optimizer.apply_gradients(zip(grads, [Tc_var]))

            # 应用物理约束（投影法）
            clipped_Tc = tf.clip_by_value(Tc_var, self.Tc_min, self.Tc_max)
            Tc_var.assign(clipped_Tc)

        return float(Tc_var[0].numpy())


# 数据生成示例
if __name__ == "__main__":
    # 初始化模型
    cstr = CSTRModel(dt=0.01)

    # 稳态初始条件
    initial_state = np.array([8.5698, 311.2639], dtype=np.float32)

    # 生成输入序列 (Tc阶跃变化)
    time_steps = 1000
    u_sequence = np.tile([10.0, 300.0, 292.0], (time_steps, 1))  # 稳态输入
    u_sequence[500:, 2] = 295.0  # 500步后改变Tc

    # 模拟生成数据
    states = cstr.generate_data(initial_state, u_sequence, time_steps)

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(states[:, 0], label='CA')
    plt.ylabel('Concentration (kmol/m³)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(states[:, 1], label='Temperature (K)')
    plt.ylabel('Temperature (K)')
    plt.xlabel('Time steps')
    plt.legend()
    plt.show()

    # MPC控制示例
    # mpc = MPCController(cstr,horizon=5)
    mpc = ImprovedMPCController(cstr,horizon=1)
    current_state = tf.constant([8.5698, 311.2639], dtype=tf.float32)
    setpoint = 320.0  # 目标温度

    # 执行MPC控制
    controlled_states = [current_state.numpy()]
    for _ in range(200):
        optimal_Tc = mpc.mpc_step(current_state, setpoint)
        u = tf.constant([10.0, 300.0, optimal_Tc], dtype=tf.float32)
        derivative = cstr.dynamics(current_state, u)
        current_state = current_state + cstr.dt * derivative
        # current_state += cstr.dt * cstr.dynamics(current_state, u)
        controlled_states.append(current_state.numpy())
        print(f"Iteration {_}, Loss: {current_state.numpy()},u:{u}")

    # 可视化控制结果
    plt.figure(figsize=(12, 6))
    plt.plot([s[1] for s in controlled_states], label='Controlled Temperature')
    plt.axhline(setpoint, color='r', linestyle='--', label='Setpoint')
    plt.ylabel('Temperature (K)')
    plt.xlabel('Time steps')
    plt.legend()
    plt.show()
