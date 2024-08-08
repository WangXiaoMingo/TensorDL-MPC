# dlmpc/__init__.py

# Import constraints
from .constraints.constraint import NonNegative, BoundedConstraint

# Import controllers
from .controllers.dl_mpc_controller import DeepLearningMPCController
from .controllers.close_loop_computation import OnlineOptimize
from .controllers.mpc_controller import MPCController

# # Import datasets
from .datasets.dataset import Dataset
from .datasets.data_loader import DataLoader
from .datasets.preprocess import WindowGenerator

# # Import dynamics
from .dynamics.NumPlant import plant_model
# from .dynamics.CSTR import CSTR
from .dynamics import SISO
from .dynamics import SISO1
from .dynamics import DelayPlant
# from .dynamics.WaterTank import WaterTank
# from .dynamics.MIMO import MIMO

# # Import layers
from .layers.normlization import MinMaxNormalization


# # Import losses
# from .losses.custom_loss import CustomLoss

# Import models
from .models.BP import BPNet
from .models.LinearRegressor import LinearRegression
from .models.Gru import GRU
from .models.Lstm import SeriesLstm, NormLstm, LSTM, ResnetLstm, SkipLstm, ResSkipLstm
from .models.Resnet_tcm import ResnetTcm

#
# # Import optimizers
from .optimizers.optimizer import optimizer
#
# Import simulation
from .simulation.simulator import SimSISO1
from .simulation.simulator import SimSISO

# Import train_models
from .train_models import train_model
from .train_models.train_model import TrainModel

# Import utils
from .utils.regression_metrics import Calculate_Regression_metrics
from .utils.plot_fig import plot_line
from .utils import loadpack
from .utils.performance_evaluation import calculate_performance_metrics

# Import version
from .version import __version__

# Define __all__ to control what is imported with "from dlmpc import *"
__all__ = [
    # Constraints
    'NonNegative',
    'BoundedConstraint',

    # Controllers
    'DeepLearningMPCController',
    'OnlineOptimize',
    'MPCController',

    # Datasets
    'Dataset',
    'DataLoader',
    'WindowGenerator',
    # Dynamics
    'plant_model',
    # 'CSTR',
    'SISO',
    'SISO1',
    # 'DelayPlant'
    # 'WaterTank',
    # 'MIMO',
    # Layers
    'MinMaxNormalization',
    # Losses
    # 'CustomLoss',

    # Models
    'BPNet',
    'LinearRegression',
    'GRU',
    'SeriesLstm',
    'NormLstm',
    'LSTM',
    'ResnetLstm',
    'SkipLstm',
    'ResSkipLstm',
    'ResnetTcm',
    # Optimizers
    'optimizer',
    # Simulation
    'SimSISO',

    # train_models
    'train_model',

    # Utils
    'Calculate_Regression_metrics',
    'plot_line',
    'calculate_performance_metrics',

    # Version
    '__version__',
]
