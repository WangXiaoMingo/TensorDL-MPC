from src.dlmpc import Dataset
from src.dlmpc import WindowGenerator
from src.dlmpc import DataLoader
if __name__ == '__main__':
    # 1. load model and generate data, (u,y)
    plant = Dataset(plant_name='SISO')
    data = plant.preprocess(num=1000)
    # plant = Dataset(plant_name='SISO1')
    # data = plant.preprocess(num=1000)
    # 2. generate Window data
    u, y = data['u'], data['y']
    input_window_dy = 2
    input_window_du = 2
    # 生成序列
    window_generator = WindowGenerator(input_window_dy, input_window_du, u, y, u_dim=1)
    # x_sequences, u_sequences, y_sequences = window_generator.generate_2D_sequences()
    x_sequences, u_sequences, y_sequences = window_generator.generate_3D_sequences()
    # 3. generate data for train, valid, test
    loader = DataLoader((x_sequences, u_sequences, y_sequences))
    data =  loader.load_data(split_seed=[0.8,0.1,0.1])
    print(data)



