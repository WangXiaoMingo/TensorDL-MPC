import tensorflow as tf

# 假设这是您在某个点设置的学习率
learning_rate = 0.1
# 创建一个 ExponentialDecay 学习率调度器
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,  # 初始学习率
    decay_steps=1,                       # 衰减步数
    decay_rate=0.99                     # 衰减率
)
# 打印学习率调度器的初始学习率和衰减步数
print(f'Initial learning rate: {learning_rate_schedule.get_config()["initial_learning_rate"]}')
print(f'Decay steps: {learning_rate_schedule.get_config()["decay_steps"]}')

# 打印学习率调度器的其他配置信息
print(learning_rate_schedule.get_config())
# 假设您想要计算在第 100 步的学习率
step = 5
current_learning_rate = learning_rate_schedule(step)
# 打印当前的学习率
print(f'Current learning rate at step {step}: {current_learning_rate:.6f}')
