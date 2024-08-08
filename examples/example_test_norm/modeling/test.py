min_val = tf.constant([-3., -1., -1.])  # 手动指定的最小值
max_val = tf.constant([1, 1., 10.])  # 手动指定的最大值
x = tf.constant(x, dtype=tf.float32)
normalization_layer = MinMaxNormalization(feature_range=(0, 1))
normalization_layer.build(x.shape)
normalization_layer.update_min_max(x)
print('test2:', normalization_layer(x_test))

a = tf.random.normal((3, 3, 3))
print(a)
normalization_layer = MinMaxNormalization(feature_range=(0, 1), min_val=min_val, max_val=max_val)
x_test_maxabs = min_max_scaler.transform(a[0])
print('x_test_maxabs1 = ', x_test_maxabs)