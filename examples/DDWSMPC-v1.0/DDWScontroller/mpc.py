import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# 自定义饱和函数，实现可微分投影
def differentiable_saturation(u, u_min, u_max, alpha=1e3):
    """可微分饱和函数实现箱式约束投影"""
    beta = (u_max + u_min) / 2.0
    return u_min + (u_max - u_min) / (1.0 + tf.exp(-alpha * (u - beta)))


class NARXModel(tf.keras.Model):
    def __init__(self, m, n, Hy, Hu, hidden_units=64):
        super().__init__()
        self.Hy = Hy
        self.Hu = Hu
        self.m = m
        self.n = n

        # 计算展平后的输入维度
        self.input_dim = Hy * m + Hu * n

        # 修改输入层结构
        self.input_layer = tf.keras.layers.Concatenate()
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation='relu', input_shape=(self.input_dim,))
        self.dense2 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.dense3 = tf.keras.layers.Dense(hidden_units // 2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(m)

    def call(self, inputs):
        """
        修正后的输入结构
        输入：
            inputs: 包含两个元素的列表 [y_hist, u_hist]
            y_hist: 历史输出序列 [batch, Hy, m]
            u_hist: 历史输入序列 [batch, Hu, n]
        """
        y_hist, u_hist = inputs

        # 展平处理
        # batch_size = tf.shape(y_hist)[0]
        y_flat = tf.reshape(y_hist, [-1, Hy*m])  # [b, Hy*m]
        u_flat = tf.reshape(u_hist, [-1, Hu*n])  # [b, Hu*n]

        # 合并输入
        x = tf.concat([y_flat, u_flat], axis=1)

        # 前向传播
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)

# 模型训练函数

def train_narx(model, X_train, y_train, epochs=50, batch_size=32):
    """修正后的训练函数"""
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')

    # 数据拆分
    val_split = 0.2
    split_idx = int(len(X_train[0]) * (1 - val_split))

    # 创建正确的输入结构
    train_data = ([X_train[0][:split_idx], X_train[1][:split_idx]], y_train[:split_idx])
    val_data = ([X_train[0][split_idx:], X_train[1][split_idx:]], y_train[split_idx:])

    history = model.fit(
        x=train_data[0],
        y=train_data[1],
        validation_data=val_data,
        batch_size=batch_size,
        epochs=epochs,
        #callbacks=[ tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True) ]
    )
    return history



class DLMPCController:
    """深度MPC控制器"""
    def __init__(self, narx_model, Hp, Hc, Q, R,
                 u_min, u_max, delta_u_min, delta_u_max):
        """
        参数：
            narx_model: 训练好的NARX模型
            Hp: 预测时域
            Hc: 控制时域
            Q: 状态跟踪权重矩阵 [Hp*m, Hp*m]
            R: 控制量权重矩阵 [Hc*n, Hc*n]
            u_min/u_max: 控制量上下限 [n]
            delta_u_min/delta_u_max: 控制增量上下限 [n]
        """
        self.narx = narx_model
        self.Hp = Hp
        self.Hc = Hc
        self.m = narx_model.m
        self.n = narx_model.n
        self.Hy = narx_model.Hy
        self.Hu = narx_model.Hu

        # 权重矩阵转换为Tensor
        self.Q = tf.constant(Q, dtype=tf.float32)
        self.R = tf.constant(R, dtype=tf.float32)

        # 约束参数
        self.u_min = tf.constant(u_min, dtype=tf.float32)
        self.u_max = tf.constant(u_max, dtype=tf.float32)
        self.delta_u_min = tf.constant(delta_u_min, dtype=tf.float32)
        self.delta_u_max = tf.constant(delta_u_max, dtype=tf.float32)

    @tf.function
    def multi_step_prediction(self, y_hist, u_hist, u_sequence):
        """多步预测生成函数"""
        y_pred = []
        current_y = tf.identity(y_hist)  # [Hy, m]
        current_u = tf.identity(u_hist)  # [Hu, n]

        # 构建全控制序列（Hp步）
        full_u = tf.concat([u_sequence,
                            tf.tile([u_sequence[-1]], [self.Hp - self.Hc, 1])], axis=0)

        # 递归预测
        for t in range(self.Hp):
            # 准备输入窗口
            y_window = current_y[-self.Hy:]  # [Hy, m]
            u_window = current_u[-self.Hu:]  # [Hu, n]

            # 预测下一时刻
            y_next = self.narx((tf.expand_dims(y_window, 0),
                               tf.expand_dims(u_window, 0)))[0]

            # 更新历史数据
            current_y = tf.concat([current_y[1:], [y_next]], axis=0)
            current_u = tf.concat([current_u[1:], [full_u[t]]], axis=0)

            y_pred.append(y_next)

        return tf.stack(y_pred)  # [Hp, m]

    def solve_mpc(self, y_hist, u_hist, y_ref,
                  max_iter=20, tol=1e-4, init_u=None):
        """
        求解MPC优化问题
        参数：
            y_hist: 历史输出序列 [Hy, m]
            u_hist: 历史输入序列 [Hu, n]
            y_ref: 参考轨迹 [Hp, m]
            init_u: 初始控制序列 [Hc, n]
        返回：
            u_opt: 最优控制序列 [Hc, n]
            y_pred: 预测输出序列 [Hp, m]
        """
        # 初始化控制序列
        if init_u is None:
            u = tf.Variable(tf.zeros([self.Hc, self.n]),
                            trainable=True, dtype=tf.float32)
        else:
            u = tf.Variable(init_u, trainable=True, dtype=tf.float32)

        # 优化循环
        opt_iter = 0
        grad_norm = tf.constant(np.inf)
        delta_J = tf.constant(np.inf)
        J_prev = tf.constant(np.inf)

        while opt_iter < max_iter and \
                grad_norm > tol and \
                delta_J > 1e-6:

            with tf.GradientTape() as tape:
                # 执行多步预测
                y_pred = self.multi_step_prediction(y_hist, u_hist, u)

                # 计算目标函数
                error = tf.reshape(y_pred - y_ref, [-1])  # [Hp*m]
                J_track = tf.tensordot(error, tf.linalg.matvec(self.Q, error),axes=1)

                u_vec = tf.reshape(u, [-1])  # [Hc*n]
                J_control = tf.tensordot(u_vec, tf.linalg.matvec(self.R, u_vec),axes=1)

                J = J_track + J_control

            # 计算梯度
            grad = tape.gradient(J, u)
            grad_flat = tf.reshape(grad, [-1])
            grad_norm = tf.norm(grad_flat)

            # 自适应步长选择（Barzilai-Borwein）
            if opt_iter == 0:
                eta = 1.0 / (grad_norm + 1e-8)
                delta_u = tf.zeros_like(u)
            else:
                delta_u = u - u_prev
                delta_g = grad_flat - grad_prev_flat
                eta = tf.tensordot(tf.reshape(delta_u, [-1]), delta_g,axes=1) / \
                      (tf.norm(delta_g) ** 2 + 1e-8)
                eta = tf.clip_by_value(eta, 1e-6, 1e3)
            u_prev = tf.identity(u)
            grad_prev_flat = tf.identity(grad_flat)
            # 梯度下降更新
            u_new = u - eta * grad

            # # 双重投影操作
            # with tf.name_scope("Projection"):
            #     # 第一步：控制量箱式约束
            #     u_proj = differentiable_saturation(u_new, self.u_min, self.u_max)
            #
            #     # 第二步：控制增量约束
            #     if opt_iter > 0:
            #         delta_u = u_proj[1:] - u_proj[:-1]
            #         delta_u_clipped = tf.clip_by_value(delta_u,
            #                                            self.delta_u_min,
            #                                            self.delta_u_max)
            #         # 重建控制序列
            #         u_proj = tf.concat([[u_proj[0]],
            #                             u_proj[1:] + delta_u_clipped], axis=0)
            #
            # # 更新变量
            # u.assign(u_proj)
            u.assign(u_new)

            # 计算目标函数变化量
            delta_J = tf.abs(J - J_prev)
            J_prev = J

            opt_iter += 1
            print(f'第{opt_iter}次,\t 损失为：{J},\t 系统输出：{y_pred[0]},\t 当前计算控制量：{u[0]}')

        return u.numpy(), y_pred.numpy()


# 数据生成函数
def generate_system_data(num_samples=5000, m=2, n=2):
    """生成训练数据（模拟二阶系统）"""
    # 系统参数
    A = np.array([[0.8, 0.1], [-0.2, 0.9]])
    B = np.array([[0.5, 0], [0, 0.3]])

    # 生成随机输入
    u_seq = np.random.uniform(-1, 1, (num_samples + 50, n))

    # 系统仿真
    y = np.zeros((num_samples + 50, m))
    for t in range(2, num_samples + 50):
        y[t] = A @ y[t - 1] + B @ u_seq[t - 1] #+ 0.1 * np.random.randn(m)

    # 构建数据集
    X_u, X_y, Y = [], [], []
    for t in range(50, num_samples + 50):
        X_y.append(y[t - 3:t])  # 3步历史输出
        X_u.append(u_seq[t - 2:t])  # 2步历史输入
        Y.append(y[t])

    return (np.array(X_y), np.array(X_u)), np.array(Y)



# 闭环仿真函数
def closed_loop_simulation(controller, num_steps=100):
    """闭环控制仿真"""
    # 初始化历史数据
    Hy = controller.narx.Hy
    Hu = controller.narx.Hu
    m = controller.m
    n = controller.n
    d = max(Hy,Hu)+1

    y_hist = np.zeros((Hy, m))
    u_hist = np.zeros((Hu, n))

    # 参考轨迹生成（正弦信号）
    t = np.arange(num_steps + controller.Hp)
    y_ref = np.column_stack([
        2 * np.sin(0.1 * t),
        np.cos(0.15 * t)
    ])
    # y_ref = np.column_stack([0.5*np.ones_like(t), 1*np.ones_like(t)])

    # 存储仿真结果
    y_actual = np.zeros((num_steps+d, m))
    u_applied = np.zeros((num_steps+d, n))

    for step in range(num_steps):
        print(f'di{step}ci')
        # 获取当前参考轨迹
        current_ref = y_ref[step:step + controller.Hp]

        # 求解MPC
        u_opt, y_pred = controller.solve_mpc(
            y_hist, u_hist, current_ref,max_iter=100, tol=1e-6,
            init_u=u_applied[step:step + controller.Hc] if step > 0 else None
        )

        # 应用第一个控制量
        u_current = u_opt[0]
        u_applied[step] = u_current

        # 系统仿真（真实系统动态）
        # 此处使用与训练数据相同的二阶系统
        A = np.array([[0.8, 0.1], [-0.2, 0.9]])
        B = np.array([[0.5, 0], [0, 0.3]])
        if step >= Hy:
            y_next = A @ y_hist[-1] + B @ u_current #+ 0.05 * np.random.randn(m)
        else:
            y_next = A @ y_hist[-1] + B @ u_current

        # 更新历史数据
        y_hist = np.concatenate([y_hist[1:], [y_next]], axis=0)
        u_hist = np.concatenate([u_hist[1:], [u_current]], axis=0)
        y_actual[step] = y_next

    return y_actual, u_applied, y_ref[:num_steps]


# 可视化函数
def plot_results(y_actual, y_ref, u_hist):
    plt.figure(figsize=(12, 8))

    # 输出跟踪
    plt.subplot(2, 1, 1)
    plt.plot(y_actual[:, 0], label='Output 1')
    plt.plot(y_ref[:, 0], '--', label='Reference 1')
    plt.plot(y_actual[:, 1], label='Output 2')
    plt.plot(y_ref[:, 1], '--', label='Reference 2')
    plt.title('Tracking Performance')
    plt.xlabel('Time Step')
    plt.ylabel('Output')
    plt.legend()

    # 控制输入
    plt.subplot(2, 1, 2)
    plt.plot(u_hist[:, 0], label='Control 1')
    plt.plot(u_hist[:, 1], label='Control 2')

    plt.title('Control Inputs')
    plt.xlabel('Time Step')
    plt.ylabel('Control Value')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 模型参数
    m = 2  # 输出维度
    n = 2  # 输入维度
    Hy = 3  # 输出历史窗口
    Hu = 2  # 输入历史窗口

    # 生成训练数据
    (X_y, X_u), Y = generate_system_data()
    print(f"Training data shapes: X_y={X_y.shape}, X_u={X_u.shape}, Y={Y.shape}")

    # 创建并训练NARX模型
    narx = NARXModel(m, n, Hy, Hu, hidden_units=64)
    train_narx(narx, (X_y, X_u), Y, epochs=100)
    print("NARX model trained.")

    # MPC参数
    Hp = 2  # 预测时域
    Hc = 2  # 控制时域

    # 权重矩阵（示例）
    Q = np.kron(np.eye(Hp), np.diag([0.1, 0.1]))  # 分块对角矩阵
    R = np.kron(0.001* np.eye(Hc), 0.001 * np.eye(n))

    # 约束条件
    u_min = np.array([-1.0, -0.8])

    u_max = np.array([1.0, 0.8])
    delta_u_min = np.array([-0.2, -0.2])
    delta_u_max = np.array([0.2, 0.2])

    # 创建MPC控制器
    mpc = DLMPCController(narx, Hp, Hc, Q, R,
                          u_min, u_max, delta_u_min, delta_u_max)

    # 运行闭环仿真
    y_actual, u_applied, y_ref = closed_loop_simulation(mpc, num_steps=100)

    # 可视化结果
    plot_results(y_actual, y_ref, u_applied)