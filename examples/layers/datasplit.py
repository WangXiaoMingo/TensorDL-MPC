import numpy as np
from typing import Union, Tuple, List


def split_timeseries_dataset(
        data: Union[np.ndarray, Tuple],
        split_ratio: List[float] = [0.8, 0.1, 0.1],
        shuffle: bool = False,
        random_seed: int = None
) -> List[Tuple]:
    """
    改进后的时序数据分割函数，支持嵌套结构

    参数:
        data: 输入数据（单个数组或嵌套元组/列表）
        split_ratio: 分割比例（总和需为1）
        shuffle: 是否打乱顺序（时序数据建议保持False）
        random_seed: 随机种子

    返回:
        分割后的数据子集列表，每个元素保持原始结构
    """
    # 验证分割比例
    if not np.isclose(sum(split_ratio), 1.0, atol=1e-3):
        raise ValueError("分割比例之和必须接近1.0")
    if min(split_ratio) <= 0:
        raise ValueError("分割比例必须全为正数")

    # 递归处理嵌套结构
    if isinstance(data, (tuple, list)):
        # 递归分割每个子结构
        split_parts = [split_timeseries_dataset(d, split_ratio, shuffle, random_seed) for d in data]
        # 重组为按分割块组织的结构
        return [tuple(parts) for parts in zip(*split_parts)]

    # 处理单个数组
    n_samples = data.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)
        data = data[indices]

    # 计算分割点
    split_points = (np.cumsum(split_ratio) * n_samples).astype(int)[:-1]

    # 执行切割
    split_data = []
    last = 0
    for point in split_points:
        split_data.append(data[last:point])
        last = point
    split_data.append(data[last:])

    return split_data


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    (hist_y, hist_u, y_ref, hist_v), control_seq = (
        (
            np.random.randn(1000, 5, 3),  # hist_y
            np.random.randn(1000, 3, 2),  # hist_u
            np.random.randn(1000, 10, 3),  # y_ref
            np.random.randn(1000, 2, 1)  # hist_v
        ),
        np.random.randn(1000, 5, 2)  # control_seq
    )

    # 构建完整数据集
    full_data = (
        (hist_y, hist_u, y_ref, hist_v),  # 输入特征组
        control_seq  # 目标序列
    )

    # 执行分割（8:1:1）
    train_data, val_data, test_data = split_timeseries_dataset(
        full_data,
        split_ratio=[0.8, 0.1, 0.1],
        shuffle=False
    )

    # 解包训练集
    (train_hist_y, train_hist_u, train_y_ref, train_hist_v), train_control = train_data
    # 解包验证集
    (val_hist_y, val_hist_u, val_y_ref, val_hist_v), val_control = val_data
    # 解包测试集
    (test_hist_y, test_hist_u, test_y_ref, test_hist_v), test_control = test_data

    # 验证维度
    print("训练集维度:")
    print(f"输入特征: {train_hist_y.shape}, {train_hist_u.shape}, {train_y_ref.shape}, {train_hist_v.shape}")
    print(f"目标序列: {train_control.shape}\n")

    print("验证集维度:")
    print(f"输入特征: {val_hist_y.shape}, {val_hist_u.shape}, {val_y_ref.shape}, {val_hist_v.shape}")
    print(f"目标序列: {val_control.shape}\n")

    print("测试集维度:")
    print(f"输入特征: {test_hist_y.shape}, {test_hist_u.shape}, {test_y_ref.shape}, {test_hist_v.shape}")
    print(f"目标序列: {test_control.shape}")