


if __name__ == '__main__':

    from src.dlmpc import Dataset
    from src.dlmpc import WindowGenerator
    from src.dlmpc import DataLoader
    from src.dlmpc import ResnetTcm
    from src.dlmpc import TrainModel
    from src.dlmpc import Calculate_Regression_metrics
    from src.dlmpc import plot_line
    from src.dlmpc import SimSISO
    from src.dlmpc import loadpack
    import seaborn as sns
    from tensorflow import keras
    from tensorflow.keras.models import load_model
    import matplotlib.pyplot as plt

    train = False

    # TODO: 1. load model and generate data, (u,y)
    plant = Dataset(plant_name='SISO',noise_amplitude=1)
    data = plant.preprocess(num=10000)

    # plant = Dataset(plant_name='SISO1')
    # data = plant.preprocess(num=1000)

    # TODO: 2. generate Window data
    u, y = data['u'], data['y']
    input_window_dy = 2
    input_window_du = 2
    dim_x = input_window_dy + input_window_du - 1  # state variable number

    #  生成序列
    window_generator = WindowGenerator(input_window_dy, input_window_du, u, y, u_dim=1)
    # x_sequences, u_sequences, y_sequences = window_generator.generate_2D_sequences()
    x_sequences, u_sequences, y_sequences = window_generator.generate_3D_sequences()

    # TODO 3. generate data for train, valid, test
    loader = DataLoader((x_sequences, u_sequences, y_sequences))
    split_seed = [0.8, 0.1, 0.1]
    (train_data, valid_data, test_data) =  loader.load_data(split_seed)
    # print(train_data['train_x_sequences'].shape)

    '''
    train_x_sequences, train_u_sequences, train_y_sequences
    valid_x_sequences, valid_u_sequences, valid_y_sequences
    test_x_sequences, test_u_sequences, test_y_sequences
    '''

    # TODO: 4. train model and save model
    my_model = ResnetTcm(num_blocks=3, kernel_size=3, nb_filters=16)  # 3 layers: input, hidden and output layers
    model = my_model.build(lstm_blocks=1, lstm_units=32, dense_units=32, dim_u=1, dim_x=dim_x, data_type='1D')
    '''data_type can select: 1D, 2D, 2DT'''
    model_name = model.name
    print(model_name)

    # train model and load best model
    if train:
        TrainModel(model,lr = 0.01,epoch=200).train_model(train_data,valid_data,show_loss=True)
        model = load_model(f'models_save/{model_name}.h5')
        model.save(f'models_save/{model_name}_predictor.h5')
    else:
        model = load_model(f'models_save/{model_name}_predictor.h5')

    model.summary()
    # TODO: predict and plot
    keras.utils.plot_model(model, to_file=f'model_fig/{model_name}.png', show_shapes=True,
                           show_layer_names=True)
    y_pred_train = model.predict([train_data['train_x_sequences'], train_data['train_u_sequences']])
    y_pred_test = model.predict([test_data['test_x_sequences'], test_data['test_u_sequences']])
    print(model.evaluate([test_data['test_x_sequences'], test_data['test_u_sequences']], test_data['test_y_sequences']))

    train_result = Calculate_Regression_metrics(y_pred_train.flatten(), train_data['train_y_sequences'].reshape(-1, 1),
                                                 label=f'{model_name}_train')
    test_result = Calculate_Regression_metrics(y_pred_test.flatten(), test_data['test_y_sequences'].reshape(-1, 1),
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
    plt.savefig(f'model_result/{model_name}_predictor_test_result.png', bbox_inches='tight', dpi=500)
    # 显示图片
    plt.show()

