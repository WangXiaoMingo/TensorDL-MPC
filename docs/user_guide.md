# User Guide

This guide provides instructions on how to use the various features of the DL_MPC library.

## **1.  Introduction**

### 1.1  Software Introduction

TensorDL-MPC toolbox is a software developed based on the Python and TensorFlow framework, which enhances the performance of traditional MPC through deep learning technology. The toolbox not only provides core functionalities such as model training, simulation testing, and parameter optimization but also supports a user-friendly interface and comprehensive documentation to reduce the barrier to entry, enabling control engineers and researchers to develop and deploy advanced control strategies more efficiently. It aims to provide advanced control strategies for industrial automation and intelligent manufacturing.

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

### 2.2 File Structure Overview

The file structure of the TensorDL-MPC toolbox is primarily divided into the following key sections:

#### 	**docs**: 

This directory contains all files related to documentation, including the user manual, API documentation, and design documentation.

- **User Manual**: Provides a detailed introduction on how to install, configure, and use the toolbox.
- **API Documentation**: Offers a detailed description and usage examples of all APIs within the toolbox.
- **Design Documentation**: Explains the design concept, architecture, and algorithm details of the toolbox.

#### 	**examples**:

 This directory contains a series of example codes and cases, demonstrating the usage methods and application scenarios of the toolbox.

- **Basic Examples**: Showcases the basic usage of the toolbox.
- **Advanced Examples**: Demonstrates how to use the toolbox to solve more complex control problems.
- **Case Studies**: Presents the application effects of the toolbox through actual cases.

#### 	**src**: 

This is the source code directory, containing the core algorithm implementations and modules of the toolbox.

- **Core Algorithms**: Implements the model predictive control algorithms and deep learning models.
- **Modular Design**: Divides the toolbox's functionalities into multiple modules for easier maintenance and expansion.

#### 	**tests**: 

This directory contains the test code, used to verify the toolbox's functionality and performance.

- **Unit Tests**: Conducts tests for each module and function to ensure code quality.
- **Integration Tests**: Tests the interaction between modules to ensure the stability of the overall functionality.
- **Performance Tests**: Evaluates the toolbox's performance indicators, such as computational speed and control algorithm performance.

	**README.md**: 

This file provides a quick overview and basic usage instructions for the toolbox.

**Quick Start**: Provides the basic steps for installing and running the toolbox.

**Contribution Guide**: Explains how to contribute code or documentation to the toolbox.
	**requirements.txt**: This file lists all the required dependencies and their versions for running the toolbox.

	**setup.py**: This script is the installation script, used for automating the installation and configuration of the toolbox.

## 3.Software description

Based on the code and software copyright statement for TensorDL-MPC: the Deep Learning-Driven Model Predictive Control (MPC) Toolbox provided by you, we can enhance the design document in the following aspects:

### **3.1. Function Description**:

#### **Core Function**:

*   Uses deep learning models to predict system dynamics and implements Model Predictive Control (MPC).
*   Supports multiple deep learning models, including BPNet, GRU, Linear Regression, SeriesLstm, NormLstm, LSTM, ResnetLstm, SkipLstm, ResSkipLstm, ResnetTcm.
*   Provides online optimization algorithms, including non-negative constraints, boundary constraints, etc.
*   Supports model training, model update, system simulation, and other functions.

#### **Target Applications**:

*   Primarily aimed at control problems of nonlinear systems, such as chemical processes, power systems, robots, etc.
*   Can be used to implement closed-loop control, that is, online correction based on the difference between actual system output and predicted output.

#### **Advantages**:

- Leverages the powerful prediction capabilities of deep learning models to improve control accuracy and robustness.
- Flexible configurability, supports multiple model and algorithm choices.
- Code modularization, easy to extend and maintain.

## **4. System Architecture**

### 4.1 **Main Modules**:

*   **Model Module**: Contains the code for building and training various deep learning models.
*   **Controller Module**: Implements the MPC control algorithm, including rolling prediction, cost function calculation, optimization algorithms, etc.
*   **Optimization Algorithm Module**: Implements various optimization algorithms, such as gradient descent, online correction, etc.
*   **System Simulation Module**: Contains the code for building and simulating system dynamic models.
*   **Data Preprocessing Module**: Contains functions for data segmentation, normalization, etc.

### 4.2 **Module Relationships**:

- The model module generates prediction models, the controller module uses the prediction models for MPC control, the optimization algorithm module optimizes the control input, the system simulation module simulates system dynamics, and the data preprocessing module preprocesses the data.
- Modules interact through data transmission, for example, the controller module passes the control input to the system simulation module, and the system simulation module passes the system output to the controller module, etc.

## **5. Technical Details**:

### **5.1 Deep Learning Models**:

*   BPNet: Multi-layer perceptron, suitable for linear relationships.
*   GRU: Gated recurrent unit, suitable for sequence prediction.
*   Linear Regression: Linear regression, suitable for linear relationships.
*   SeriesLstm, NormLstm, LSTM, ResnetLstm, SkipLstm, ResSkipLstm: LSTM variants, suitable for sequence prediction, and introduce residual connections, skip connections, etc.
*   ResnetTcm: A model combining residual networks and causal convolutions, suitable for sequence prediction.

### 5.2 **MPC Control Algorithm**:

*   Rolling Prediction: Based on the current state and prediction model, stepwise predicts future states and outputs.
*   Cost Function: Used to measure the difference between predicted output and actual output.
*   Optimization Algorithm: Used to find the control input that minimizes the cost function.

### **5.3 Optimization Algorithms**:

*   Non-negative constraints: Ensures that the control input is non-negative.
*   Boundary constraints: Ensures that the control input is within a specified range.


### **5.4 Usage Examples**:

*   Example code demonstrates how to use TensorDL-MPC for MPC control, including initializing the system, training the model, performing MPC control, simulating system dynamics, etc.



#### **5.4.1 . Usage Steps**

**1. Data Preprocessing**

* **Load Data**: Use the `Dataset` class to generate training data from a simulation system or real data. For example, use the `SimSISO` class to generate training data for a SISO system.

* **Window Generation**: Use the `WindowGenerator` class to split the data into windows, each containing system states, control inputs, and outputs.

* **Data Loading**: Use the `DataLoader` class to split the window data into training sets, validation sets, and test sets.

  **2. Model Training**

* **Choose Model**: Choose an appropriate deep learning model for training. TensorDL-MPC provides various models, such as BPNet, GRU, LSTM, etc.

* **Model Building**: Use the selected model class to build the model and set model parameters, such as the number of hidden layer units, learning rate, etc.

* **Model Training**: Use the `TrainModel` class to train the model and pass the training data and validation data to the model.

* **Model Saving**: Save the trained model to a file for later use.
  **3. MPC Control Process**

* **Initialize System State**: Set the initial system state and control input.

* **Load Model**: Load the trained deep learning model.

* **MPC Control Loop**:
    * Create an MPC controller instance using the `MPCController` class or `DeepLearningMPCController` class.
    * Optimize control input using the MPC controller and obtain the optimized control input sequence.
    * Update the system state based on the optimized control input.
    * Calculate the tracking error and use an online correction algorithm for correction.
    * Repeat the above steps until the specified control cycle is reached or a stopping condition is met.
    **4. Performance Evaluation**
    
* Use the `calculate_performance_metrics` function to calculate performance metrics of the MPC control process, such as ISE, IAE, overshoot, etc.

* Visualize the results of the MPC control process, such as system output, reference trajectory, and control input.
**Code Example**
The following is an example code using the BPNet model and MPC control:
```python
from dlmpc import SimSISO, WindowGenerator, DataLoader, BPNet, TrainModel
from dlmpc import DeepLearningMPCController
# Data Preprocessing
plant = SimSISO(plant_name='SISO', noise_amplitude=1)
data = plant.preprocess(num=1000)
u, y = data['u'], data['y']
input_window_dy = 2
input_window_du = 2
window_generator = WindowGenerator(input_window_dy, input_window_du, u, y, u_dim=1)
x_sequences, u_sequences, y_sequences = window_generator.generate_3D_sequences()
loader = DataLoader((x_sequences, u_sequences, y_sequences))
split_seed = [0.8, 0.1, 0.1]
(train_data, valid_data, test_data) = loader.load_data(split_seed)
# Model Training
my_model = BPNet(hidden_blocks=2)
model = my_model.build(units=32, dim_u=1, dim_x=input_window_dy + input_window_du - 1, data_type='1D')
TrainModel(model, lr=0.01, epoch=200).train_model(train_data, valid_data, show_loss=True)
model.save(f'models_save/{model.name}_predictor.h5')
# MPC Control
mpc_controller = DeepLearningMPCController(model, predict_horizon=4, control_horizon=2, Q=np.eye(4) * 0.1, R=np.eye(2) * 0.01, ly=input_window_dy, lu=input_window_du - 1, dim_u=1, du_bounds=[-1, 1], u_bounds=[-5, 5], opt=optimizer(optimizer_name='sgd', learning_rate=0.1))
state_y = tf.constant([[1], [1.2]], dtype=tf.float32)
state_u = tf.constant([[0.1]], dtype=tf.float32)
u0 = tf.constant([0.2], dtype=tf.float32)
y_ref = 10
for i in range(50):
    parameter = mpc_controller.optimize(error=0, state_y=state_y, state_u=state_u, y_ref=y_ref, iterations=100, tol=1e-6)
    u0 = parameter['u0']
    plant_output = plant.plant(np.append(tf.squeeze(state_u), u0), tf.squeeze(state_y))
    state_y, state_u, error = mpc_controller.estimate(u0, plant_output)
    y_ref = 10 - 2 * i
```
**Note**:
* Please adjust the parameters in the code according to your specific problem, such as model structure, MPC parameters, etc.
* You can try using other models or MPC control strategies to achieve better control performance.
* You can use other functions provided by TensorDL-MPC, such as model updating, fault detection, etc., to enhance the robustness and reliability of the MPC control system.



