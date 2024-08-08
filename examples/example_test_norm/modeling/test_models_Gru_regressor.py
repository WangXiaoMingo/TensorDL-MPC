


if __name__ == '__main__':

    from src.dlmpc import WindowGenerator
    from src.dlmpc import DataLoader
    from src.dlmpc import GRU
    from src.dlmpc import TrainModel
    from src.dlmpc import Calculate_Regression_metrics
    from src.dlmpc import plot_line
    from src.dlmpc import MinMaxNormalization
    from src.dlmpc import SimSISO
    from src.dlmpc import loadpack
    import seaborn as sns
    from tensorflow import keras
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    train = True
    norm1 = True

    data = pd.read_excel('../data/KTT_7.xlsx', header=0, index_col='时间')
    # data = data[['下料量','转速','进风管压力','窑尾']].iloc[:2000]
    data = data[['下料量', '窑尾']]
    # print(data.describe())
    if norm1:
        min_val_u = tf.constant([0.])    # 手动指定的最小值
        max_val_u = tf.constant([40.])   # 手动指定的最大值
        min_val_y = tf.constant([0.])  # 手动指定的最小值
        max_val_y = tf.constant([1000.])  # 手动指定的最大值
        normalization_layer_u = MinMaxNormalization(feature_range=(-1, 1), min_val=min_val_u, max_val=max_val_u)
        normalization_layer_y = MinMaxNormalization(feature_range=(-1, 1), min_val=min_val_y, max_val=max_val_y)
        u = normalization_layer_u(np.array(data['下料量']))
        y = normalization_layer_y(np.array(data['窑尾']))
    else:
        u, y = data['下料量'], data['窑尾']

    # TODO: 2. generate Window data
    input_window_dy = 2
    input_window_du = 2
    dim_x = input_window_dy + input_window_du - 1  # state variable number

    #  生成序列
    window_generator = WindowGenerator(input_window_dy, input_window_du, u, y, u_dim=1)
    x_sequences, u_sequences, y_sequences = window_generator.generate_3D_sequences()

    # TODO 3. generate data for train, valid, test
    loader = DataLoader((x_sequences, u_sequences, y_sequences))
    split_seed = [0.8, 0.1, 0.1]
    (train_data, valid_data, test_data) = loader.load_data(split_seed)
    # print(train_data['train_x_sequences'].shape)

    '''
    train_x_sequences, train_u_sequences, train_y_sequences
    valid_x_sequences, valid_u_sequences, valid_y_sequences
    test_x_sequences, test_u_sequences, test_y_sequences
    '''

    # TODO: 4. train model and save model

    model = GRU().build(gru_blocks=1, gru_units=32, dense_units=32, dim_u=1, dim_x=dim_x, data_type='1D')
    '''data_type can select: 1D, 2D, 2DT'''
    model_name = model.name
    print(model_name)

    # train model and load best model
    if train:
        if norm1:
            TrainModel(model,lr = 0.0001,epoch=200).train_model(train_data,valid_data,show_loss=True) # 全部数据
        else:
            TrainModel(model, lr=0.0001, epoch=200).train_model(train_data, valid_data, show_loss=True)
        model = load_model(f'models_save/{model_name}.h5')
        model.save(f'../models_save/{model_name}_predictor.h5')
    else:
        model = load_model(f'../models_save/{model_name}_predictor.h5')

    model.summary()
    # TODO: predict and plot
    keras.utils.plot_model(model, to_file=f'../model_fig/{model_name}.png', show_shapes=True,
                           show_layer_names=True)

    if norm1:
        y_pred_train = normalization_layer_y.inverse_call(
            model.predict([train_data['train_x_sequences'], train_data['train_u_sequences']])).numpy().reshape(-1, 1)
        y_pred_test = normalization_layer_y.inverse_call(
            model.predict([test_data['test_x_sequences'], test_data['test_u_sequences']])).numpy().reshape(-1, 1)
        print(model.evaluate([test_data['test_x_sequences'], test_data['test_u_sequences']],
                             test_data['test_y_sequences']))

        train_result = Calculate_Regression_metrics(y_pred_train,
                                                    normalization_layer_y.inverse_call(
                                                        train_data['train_y_sequences'].reshape(-1, 1)),
                                                    label=f'{model_name}_train')
        test_result = Calculate_Regression_metrics(y_pred_test, normalization_layer_y.inverse_call(
            test_data['test_y_sequences'].reshape(-1, 1)),
                                                   label=f'{model_name}_test')
        figure_property = {'title': model_name, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
        plot_line(y_pred_test, normalization_layer_y.inverse_call(test_data['test_y_sequences'].reshape(-1, 1)),
                  figure_property)

    else:
        y_pred_train = model.predict([train_data['train_x_sequences'], train_data['train_u_sequences']])
        y_pred_test = model.predict([test_data['test_x_sequences'], test_data['test_u_sequences']])
        print(model.evaluate([test_data['test_x_sequences'], test_data['test_u_sequences']],
                             test_data['test_y_sequences']))

        train_result = Calculate_Regression_metrics(y_pred_train.flatten(),
                                                    train_data['train_y_sequences'].reshape(-1, 1),
                                                    label=f'{model_name}_train')
        test_result = Calculate_Regression_metrics(y_pred_test.flatten(),
                                                   test_data['test_y_sequences'].reshape(-1, 1),
                                                   label=f'{model_name}_test')

        figure_property = {'title': model_name, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
        plot_line(y_pred_test.flatten(), test_data['test_y_sequences'].reshape(-1, 1), figure_property)
        
    print('train\n ', train_result)

    print('test:\n', test_result)
    # 设置Seaborn样式
    sns.set_style("whitegrid")
    # 创建一个Matplotlib图
    fig, ax = plt.subplots(figsize=(12, 2))
    # 移除坐标轴
    ax.axis('off')
    ax.set_title(f'{model_name}_predictor test result.png', fontsize=16, pad=2)  # pad参数可以调整标题与表格之间的距离
    # 将DataFrame转换为表格
    tab = ax.table(cellText=test_result.values, colLabels=test_result.columns, rowLabels=test_result.index, loc='center')
    # 可以为表格添加样式
    tab.auto_set_font_size(True)
    tab.set_fontsize(12)
    tab.scale(1.0, 1.0)
    # 保存图片
    plt.savefig(f'../model_result/{model_name}_predictor_test_result.png', bbox_inches='tight', dpi=500)
    # 显示图片
    plt.show()

    data = pd.DataFrame([y_pred_test.flatten(), test_data['test_y_sequences'].reshape(-1)]).T
    data.columns = ['GRU', 'test_real']
    # data.to_excel('../model_result/GRU.xlsx')

