from src.dlmpc import Dataset
if __name__ == '__main__':
    # load model and generate data, (u,y)
    plant = Dataset(plant_name='SISO')
    data = plant.preprocess(num=1000)

    plant = Dataset(plant_name='SISO1')
    data = plant.preprocess(num=1000)


