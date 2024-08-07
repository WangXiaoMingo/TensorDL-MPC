## **1.  Introduction**

### **1. Overview**

The TensorDL-MPC toolbox is a Python-based software developed using the TensorFlow framework. It leverages deep learning techniques to enhance the performance of traditional Model Predictive Control (MPC). The toolbox not only provides core functionalities such as model training, simulation testing, and parameter optimization but also offers a user-friendly interface and comprehensive documentation to facilitate efficient development and deployment of advanced control strategies for industrial automation and intelligent manufacturing.

TensorDL-MPC: Deep Learning-Driven Model Predictive Control Toolbox features include:

- **Deep Learning Integration**: Supports multiple deep learning models, capable of handling high-dimensional and nonlinear data, improving control accuracy.
- **Model Predictive Control:** Utilizes advanced MPC algorithms combined with deep learning models to achieve accurate prediction and optimal control of future system states.
- **User-friendly Interface**: Provides concise and clear APIs for users to quickly get started and customize control strategies.
- **Simulation and Testing**: Built-in simulation environment allows users to test control strategies in a safe environment and evaluate performance.
- **Simulation Cases**: Includes multiple simulation cases to help users understand the application and effects of the toolbox.
- **Documentation and Support**: Provides documentation and technical support to ensure users can fully utilize the toolbox.
- **Modular Design**: Uses a modular development approach for easy feature expansion and maintenance.

TensorDL-MPC toolbox is suitable for various industrial control scenarios, including but not limited to:

- **Chemical Process Control**: Achieves precise control in chemical reactors, distillation towers, and other chemical equipment.
- **Manufacturing Process Optimization**: Optimizes production processes on the production line to improve product quality and production efficiency.
- **Energy Management**: Manages and optimizes energy effectively in power systems, energy distribution networks.

- **Autonomous Driving Vehicles**: Conducts path planning and dynamic decision-making in autonomous driving systems.

### 1.2 Software and Hardware Environment

Software Environment (TensorDL-MPC toolbox development and supports the following software environments):

- **Operating System**: Windows 10 (64-bit) Professional or higher, Linux (recommended Ubuntu 18.04 LTS and above), macOS (recommended Catalina 10.15 and above).
- **Programming Language**: Python 3.6 and above, ensuring compatibility with the TensorFlow framework.
- **Development Framework**: TensorFlow 1.13 and above, for building and training deep learning models.
- **Development Tools**: Recommended to use Anaconda or Miniconda for environment management, Visual Studio Code or PyCharm as the integrated development environment (IDE).
- **Other Dependencies**: NumPy, SciPy, Pandas, and other common scientific computing libraries. Specific dependency versions can be found in the requirements.txt file.

Hardware Environment To achieve optimal performance, TensorDL-MPC toolbox recommends the following hardware configuration:

- **Processor**: Intel Core i7 or higher CPU with multi-core and high clock frequency.
- **Memory**: At least 8GB RAM, recommended 16GB or higher to support complex model training.
- **Storage**: SSD hard drive with at least 256GB of available storage space to ensure sufficient read-write speed and storage capacity.
- **Graphics Processor**: Recommended to use NVIDIA series graphics cards with at least 4GB of memory, supporting CUDA and cuDNN to accelerate deep learning model training.

## 2. Software Function

### 2.1 Software Overview

TensorDL-MPC toolbox integrates deep learning technology with model predictive control algorithms to provide an efficient, accurate, and user-friendly control strategy development platform. Its main functions include: deep learning integration, advanced MPC algorithms, user-friendly interface, simulation and testing environment, simulation cases, documentation and technical support, and modular design. 

The structure of TensorDL-MPC: Deep Learning-Driven Model Predictive Control Toolbox is shown in Figure 2.1.

![img](file:///C:/Users/ADMINI~1/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

Figure 2.1 Structure of TensorDL-MPC: Deep Learning-Driven Model Predictive Control Toolbox



### **2.2 Functional Modules**

TensorDL-MPC employs a modular design, incorporating the following key functional modules:

#### **2.2.1 Dataset Module (datasets**)

The dataset module serves as the foundation for data handling, encompassing data loading and preprocessing. It supports various data formats, and offers flexible preprocessing tools like data cleaning, normalization, and batching. 

#### **2.2.2 Dynamics Module (dynamics**)

The dynamics module is crucial for modeling the behavior of physical systems. It provides multiple modeling approaches, such as differential equations, transfer functions, and state-space models. Users can select an appropriate method based on the system's characteristics and customize model parameters for accurate system representation.

#### **2.2.3 Neural Network Layer Structure Module (layers**)

This module offers various neural network layer structures, including convolutional layers, recurrent layers, and fully connected layers. Users can choose suitable layer structures based on the requirements of their deep learning models and customize layer parameters like neuron count and activation functions.

#### **2.4 Loss Function Module (losses**)

The loss function module provides diverse loss functions to measure the discrepancy between model predictions and true values. Users can select appropriate loss functions based on their learning tasks and customize parameters like weight coefficients.

#### *2.5 Model Construction Module (model**)

The model construction module offers various methods for building deep learning models, including Sequential models, Functional API, and Model subclassing. Users can choose the most suitable method based on their model requirements and customize model structure, layer parameters, and loss functions.

#### **2.6 Model Training Module (train_models**)

The model training module handles the training process, supporting multiple optimization algorithms like Adam, SGD, and RMSprop. It also provides flexible training strategies like learning rate decay and early stopping. Users can select appropriate optimization algorithms and training strategies based on their learning tasks and customize training parameters like learning rate and iteration count.

#### **2.7 Controller Design Module (controllers**)

The controller design module implements Model Predictive Control algorithms, including objective function setup, rolling prediction, online optimization, and online correction. Users can select suitable control algorithms based on their control tasks and customize controller parameters like prediction horizon and control horizon.

#### **2.8 Constraint Module (constraints**)

The constraint module enables the setting of constraints for the controller, such as input constraints, output constraints, and state constraints. Users can define appropriate constraints based on their control tasks to ensure the controller outputs safe and stable control strategies.

#### **2.9 Optimizer Module (optimizers**)

The optimizer module offers various optimization algorithms like Adam, SGD, and RMSprop. Users can select suitable optimization algorithms based on their optimization tasks and customize optimizer parameters like learning rate and iteration count.

#### **2.10 Simulation Module (simulation**)

The simulation module facilitates the simulation and validation of control strategies. It generates system state trajectories based on the dynamic system model and controller outputs, evaluating performance metrics like steady-state error and dynamic response.

#### **2.11 Utility Function Module (utils**)

The utility function module provides various tools and utilities for data visualization, performance evaluation, and other tasks, facilitating easier data handling, computation, and development.

## **3. Software Architecture**

The software architecture of the TensorDL-MPC toolbox is depicted in Figure 3-1.

```
                                       ┌──────────────┐
                                       │    User      │
                                       └──────────────┘
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │    API       │
                                       └──────────────┘
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │    Core      │
                                       │    Module    │
                                       └──────────────┘
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │    Functional│
                                       │    Modules   │
                                       └──────────────┘
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │    Dependency│
                                       │    Libraries │
                                       └──────────────┘
```
## **4. Design Principles**

- The design of the TensorDL-MPC toolbox adheres to the following principles:
- **Modular Design**: Dividing the toolbox functionality into modules for easier maintenance and expansion.
- **Extensibility**: Providing interfaces and tools for users to customize modules and functionalities.
- **Usability**: Offering user-friendly APIs and documentation to lower the barrier to entry.
- **High Performance**: Employing efficient algorithms and parallel computing techniques to enhance computational speed and control algorithm performance.
- **Reliability**: Conducting rigorous testing to ensure software quality and stability.

## **5. Conclusion**

The TensorDL-MPC toolbox is a powerful, user-friendly, and extensible deep learning-driven Model Predictive Control toolbox. It empowers control engineers and researchers to efficiently develop and deploy advanced control strategies, providing cutting-edge control solutions for industrial automation and intelligent manufacturing.