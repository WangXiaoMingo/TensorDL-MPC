import tensorflow as tf
def create_custom_loss(model, lambda1=0.1, lambda2=0.1):
    # 定义损失函数（接受y_true和y_pred，同时隐式使用输入数据）
    def custom_loss(y_true, y_pred):
        # MSE损失
        mse_loss = tf.reduce_mean(tf.reduce_sum((y_true - y_pred) ** 2, axis=-1))

        # 梯度平滑项
        with tf.GradientTape() as tape:
            tape.watch(model.input)
            predictions = model(model.input)
        jacobian = tape.jacobian(predictions, model.input)  # 形状 (batch, control_dim, state_dim)
        grad_loss = tf.reduce_mean(tf.reduce_sum(jacobian ** 2, axis=[1, 2]))

        # 控制量平滑项（假设输入为时间序列）
        # 注意：需要确保输入数据是时间序列（batch_size, time_steps, state_dim）
        # 此处假设输入已按时间步展开，需在训练时重组为时间序列
        batch_size = tf.shape(y_true)[0] #// time_steps  # 计算原始batch_size
        y_pred_reshaped = tf.reshape(y_pred, (batch_size, time_steps, control_dim))
        diff = y_pred_reshaped[:, 1:, :] - y_pred_reshaped[:, :-1, :]
        smooth_loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=[1, 2]))

        # 总损失
        return mse_loss + lambda1 * grad_loss + lambda2 * smooth_loss

    return custom_loss



lambda1 = 0.1  # 梯度平滑项系数
lambda2 = 0.1  # 控制量平滑项系数

# 创建自定义损失函数
custom_loss = create_custom_loss(model, lambda1, lambda2)

# 编译模型（使用自定义损失）
model.compile(optimizer='adam', loss=custom_loss)

# 假设输入数据已按时间步展开为 (num_samples * time_steps, state_dim)
history = model.fit(
    x=states_flat,
    y=controls_flat,
    batch_size=32,
    epochs=50,
    validation_split=0.2
)




