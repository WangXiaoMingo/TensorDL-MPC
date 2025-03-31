import numpy as np


def generate_mimo_data(data_y, data_u, data_v=None, Hy=3, Hu=3, Hd=3):
    """
    生成MIMO系统数据

    参数:
        data_y (list or array): 输出时间序列数据
        data_u (list or array): 输入时间序列数据
        data_v (list or array, optional): 扰动时间序列数据，默认为 None
        Hy (int): 输出历史数据的窗口大小，默认为 3
        Hu (int): 输入历史数据的窗口大小，默认为 3
        Hd (int): 扰动历史数据的窗口大小，默认为 3

    返回:
        tuple: 包含输入特征 (hist_y, hist_u, hist_d) 和目标输出 next_y
    """
    # 类型检查
    if not isinstance(data_y, (list, np.ndarray)) or not isinstance(data_u, (list, np.ndarray)):
        raise TypeError("data_y 和 data_u 必须是列表或 NumPy 数组")
    if data_v is not None and not isinstance(data_v, (list, np.ndarray)):
        raise TypeError("data_v 必须是列表或 NumPy 数组，或者为 None")

    # 长度检查
    N = len(data_y)
    if len(data_u) != N:
        raise ValueError("data_y 和 data_u 的长度必须相同")
    if data_v is not None and len(data_v) != N:
        raise ValueError("data_v 的长度必须与 data_y 和 data_u 相同")

    # 计算起始索引
    start = max(Hy, Hu) if data_v is None else max(Hy, Hu, Hd)
    if start >= N:
        raise ValueError("时间序列长度不足以生成所需的历史数据")

    # 使用 NumPy 滑动窗口生成历史数据
    hist_y = np.array([data_y[t - Hy:t] for t in range(start, N)])
    hist_u = np.array([data_u[t - Hu:t] for t in range(start, N)])
    next_y = np.array(data_y[start:])

    # 如果 data_v 存在，则生成扰动历史数据
    if data_v is not None:
        hist_d = np.array([data_v[t - Hd:t] for t in range(start, N)])
        return (hist_y, hist_u, hist_d), next_y
    else:
        return (hist_y, hist_u), next_y


def generate_policy_data(data_y, data_u, data_v=None, Hy=3, Hu=3, Hd=3, Hp=10, Hc=5):
    """
    生成包含扰动数据的MIMO系统训练数据

    参数:
        data_y (np.ndarray): 状态序列 (N, n_y)
        data_u (np.ndarray): 控制序列 (N, n_u)
        data_v (np.ndarray): 扰动序列 (N, n_v)
        Hy (int): 状态历史时域
        Hu (int): 控制历史时域
        Hd (int): 扰动历史时域
        Hp (int): 预测时域
        Hc (int): 控制时域

    返回:
        tuple: (hist_y, hist_u, hist_v, y_ref), (control_seq, y_target)
    """
    # 数据维度验证
    N, n_y = data_y.shape
    N_u, n_u = data_u.shape
    if N != N_u:
        raise ValueError("data_y和data_u的时间长度必须一致")

    # 扰动数据存在性检查
    has_disturbance = data_v is not None
    if has_disturbance:
        N_v, n_v = data_v.shape
        if N_v != N:
            raise ValueError("data_v的时间长度必须与其他数据一致")

    # 计算有效样本范围
    start = max(Hy, Hu, Hd) if has_disturbance else max(Hy, Hu)
    end = N - max(Hp, Hc)
    if start >= end:
        raise ValueError(f"数据长度不足，需要至少{start + max(Hp, Hc)}个样本")

    # 预分配内存
    num_samples = end - start
    hist_y = np.zeros((num_samples, Hy, n_y))
    hist_u = np.zeros((num_samples, Hu, n_u))
    hist_v = np.zeros((num_samples, Hd, n_v)) if has_disturbance else None
    y_ref = np.zeros((num_samples, Hp, n_y))
    control_seq = np.zeros((num_samples, Hc, n_u))


    # 滑动窗口生成
    for i in range(num_samples):
        t = start + i

        # 历史状态窗口 (t-Hy 到 t-1)
        hist_y[i] = data_y[t - Hy: t]

        # 历史控制窗口 (t-Hu 到 t-1)
        hist_u[i] = data_u[t - Hu: t]

        # 历史扰动窗口 (t-Hd 到 t-1)
        if has_disturbance:
            hist_v[i] = data_v[t - Hd: t]

        # 未来参考轨迹 (t 到 t+Hp-1)
        y_ref[i] = data_y[t: t + Hp]

        # 未来控制序列 (t 到 t+Hc-1)
        control_seq[i] = data_u[t: t + Hc]


    # 构造返回元组
    inputs = (hist_y, hist_u, y_ref)
    if has_disturbance:
        inputs += (hist_v,)

    targets = control_seq

    return inputs, targets


if __name__ == "__main__":
    num_samples = 20
    y = np.random.rand(num_samples + 10)  # 输出时间序列
    u = np.random.rand(num_samples + 10)  # 输入时间序列
    v = np.random.rand(num_samples + 10)  # 扰动时间序列

    # 调用函数生成数据
    Hy, Hu, Hd = 3, 2, 3
    (X_y, X_u, X_d), Y = generate_mimo_data(y, u, v, Hy, Hu, Hd)
    print(X_y.shape, X_u.shape, X_d.shape, Y.shape)

    # 测试 data_v 为 None 的情况
    (X_y, X_u), Y = generate_mimo_data(y, u, None, Hy, Hu, None)
    print(X_y.shape, X_u.shape, Y.shape)




    # 生成示例数据
    N = 1000
    n_y = 3  # 温度、压力、流量
    n_u = 2  # 阀门开度、转速
    n_v = 1  # 环境温度

    # 模拟数据
    data_y = np.column_stack([
        np.sin(np.linspace(0, 10 * np.pi, N)),  # 温度
        np.cos(np.linspace(0, 5 * np.pi, N)),  # 压力
        np.random.normal(0, 0.5, N)  # 流量
    ])

    data_u = np.column_stack([
        np.clip(np.cumsum(np.random.randn(N)), -1, 1),  # 阀门开度
        np.sin(np.linspace(0, 8 * np.pi, N))  # 转速
    ])

    data_v = np.random.randn(N, n_v)  # 环境温度扰动

    # 生成训练数据
    (hist_y, hist_u, y_ref, hist_v), control_seq = generate_policy_data(
        data_y=data_y,
        data_u=data_u,
        data_v=data_v,
        Hy=5,
        Hu=3,
        Hd=2,
        Hp=10,
        Hc=5
    )

    # 数据验证
    print("输入数据维度:")
    print(f"hist_y: {hist_y.shape}")
    print(f"hist_u: {hist_u.shape}")
    print(f"hist_v: {hist_v.shape}")
    print(f"y_ref: {y_ref.shape}")
    print("\n目标数据维度:")
    print(f"control_seq: {control_seq.shape}")

