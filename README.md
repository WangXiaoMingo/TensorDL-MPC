# DL-MPC: A toolbox for deep learning-based nonlinear model predictive control

## **1. Overview**

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

## 2 Software and Hardware Environment

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

# 2. Installation Guide

This guide walks you through the process of installing the DL_MPC library.

## Method 1: Quick installation

### 1. Install Anaconda and PyCharm

- First, you need to install Anaconda and PyCharm. Follow the installation guides on their respective websites to complete the installation.

### 2. Create an Anaconda Environment

1.  Open Anaconda Prompt.
2.  Use the following command to create a new virtual environment (e.g., `TensorDL-MPC_env`)(python 3.8+):

```bash
conda create --name TensorDL-MPC_env python=3.8
```

3.  Activate the virtual environment:

```bash
conda activate TensorDL-MPC_env
```

### **3. Install Dependencies**

In the activated virtual environment, use the following command to install TensorFlow (2.5+), you can also use: requirements.txt

```bash
pip install tensorflow-gpu==2.9.0
```

If you want to install other dependencies, such as Scikit-learn, use the following command:
Note:
If the pandas version is too high, there may be errors, such as the part where 
pd.set_option('precision', 4) is used. 
In this case, you can comment this out.

```bash
pip install scikit-learn  #pip install scikit-learn==1.2.2
pip install pandas==1.3.5
pip install matplotlib
pip install openpyxl
```

### **4. Create a PyCharm Project**

- Open PyCharm.
- Click "Create New Project".
- Select "Python" as the project type.
- Choose "Existing interpreter" and select your Anaconda virtual environment.

### 5. Clone the Code Repository

In PyCharm, use the Git repository browser to clone the TensorDL-MPC code repository into your project directory.
```bash
https://github.com/WangXiaoMingo/TensorDL-MPC
```

### 6. Configure PyCharm

- In PyCharm, go to "File" -> "Settings" (or press Ctrl+Alt+S).
- Select "Project: TensorDL-MPC" -> "Python Interpreter".
- In the "Python Interpreter" page, click "Add" to add your Anaconda virtual environment.

### 7. Run the Example

- In PyCharm, find the example script file (e.g., xxx.py).
- Right-click on the file and select "Run 'xxx.py'".

### 8. Notes

*   Virtual Environment: Make sure you select the correct Anaconda virtual environment when creating the PyCharm project.
*   Dependencies: Make sure you have configured the correct Python interpreter and dependencies in PyCharm.



# Method 2: Install from scratch

### **1. Install Anaconda**

- Visit the [Anaconda website](https://www.anaconda.com/products/distribution) to download the Anaconda installation program.
- Run the installation program and follow the prompts to install. It is recommended to select the "Just me (local user only)" option to install Anaconda in your user directory.

## **2. Open Command Line**

- On Windows, press Win + R, enter `cmd`, and press Enter to open the command line.
- On macOS or Linux, open the terminal application.

### **3. Create an Anaconda Virtual Environment**

- In the command line, enter the following command to create a new virtual environment (e.g., `TensorDL-MPC_env`):

```bash
conda create --name TensorDL-MPC_env python=3.8
```

- Activate the virtual environment:

```bash
conda activate TensorDL-MPC_env
```

### **4. Install Dependencies**

In the activated virtual environment, install TensorFlow and NumPy using the following command:

```bash
pip install tensorflow-gpu==2.9.0
```

If you want to install other dependencies, such as Scikit-learn, use the following command:

```bash
pip install scikit-learn==1.2.2
pip install pandas==1.3.5
pip install matplotlib
pip install openpyxl
```

### **5. Clone the Code Repository**

In the command line, use the following command to clone the TensorDL-MPC code repository into your project directory:

```bash
git clone https://github.com/WangXiaoMingo/TensorDL-MPC.git
```

### **6. Install Project Dependencies**

In the TensorDL-MPC code repository directory, use the following command to install project dependencies:

```bash
pip install -r requirements.txt
```

### **7. Run the Example**

1.  In PyCharm, find the example script file (e.g., `xxx.py`).
2.  Right-click on the file and select "Run 'xxx.py'".

### **8. Notes**

*   **Virtual Environment**: Make sure you activate the correct Anaconda virtual environment before installing dependencies and running the code.
*   **pip**: Ensure that pip is installed in your Python environment.
*   **PyCharm**: If you want to use PyCharm, make sure you have installed and opened PyCharm.

# Method 3: setup.py Install 

### **1. Activate Anaconda Virtual Environment**

If you have not already activated an Anaconda virtual environment, please follow these steps:

- Open Command Prompt (CMD).
- Enter the following command to create a new virtual environment (e.g., `TensorDL-MPC_env`) and activate it:

```bash
conda create --name TensorDL-MPC_env python=3.8
conda activate TensorDL-MPC_env
```

### **2. Clone GitHub Repository**

1. In Command Prompt, enter the following command to clone the TensorDL-MPC repository to your local machine:

```bash
git clone https://github.com/WangXiaoMingo/TensorDL-MPC.git
```

2. Switch to the cloned repository directory:

```bash
cd TensorDL-MPC
```

### **3. Install Project Dependencies**

1. In the repository directory, enter the following command to install the dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

2. Enter the following command to install the project itself using the `setup.py` file:

```bash
python setup.py install
```

### **4. Verify Installation**

1. Enter the following command to verify that the `setup.py` has installed the project:

```bash
python -c "import src.dlmpc"
```

If the command does not produce any errors, it indicates that the project has been installed successfully.

### **5. Run Examples**

1. Enter the following command to run an example script from the project (if available):

```bash
python xxx.py
```

### **6. Considerations**

- Ensure that you have Anaconda and Python installed.
- Ensure that pip is installed in your Anaconda environment.
- If you are using Python 3.10 or a higher version and pip is not installed, install pip first:

```bash
python -m pip install --upgrade pip
```

- If you encounter any issues during installation, please check your `setup.py` file and `requirements.txt` file to ensure they are correct.
  By following these steps, you should be able to fully install and configure the TensorDL-MPC project from the command line on a Windows environment. If you encounter any problems during the process, make sure you have activated the correct Anaconda virtual environment in your Command Prompt.


##  **3. System Architecture**

### 3.1 **Main Modules**:

*   **Model Module**: Contains the code for building and training various deep learning models.
*   **Controller Module**: Implements the MPC control algorithm, including rolling prediction, cost function calculation, optimization algorithms, etc.
*   **Optimization Algorithm Module**: Implements various optimization algorithms, such as gradient descent, online correction, etc.
*   **System Simulation Module**: Contains the code for building and simulating system dynamic models.
*   **Data Preprocessing Module**: Contains functions for data segmentation, normalization, etc.

### 3.2 **Module Relationships**:

- The model module generates prediction models, the controller module uses the prediction models for MPC control, the optimization algorithm module optimizes the control input, the system simulation module simulates system dynamics, and the data preprocessing module preprocesses the data.
- Modules interact through data transmission, for example, the controller module passes the control input to the system simulation module, and the system simulation module passes the system output to the controller module, etc.

## **4. Technical Details**:

### **4.1 Deep Learning Models**:

*   BPNet: Multi-layer perceptron, suitable for linear relationships.
*   GRU: Gated recurrent unit, suitable for sequence prediction.
*   Linear Regression: Linear regression, suitable for linear relationships.
*   SeriesLstm, NormLstm, LSTM, ResnetLstm, SkipLstm, ResSkipLstm: LSTM variants, suitable for sequence prediction, and introduce residual connections, skip connections, etc.
*   ResnetTcm: A model combining residual networks and causal convolutions, suitable for sequence prediction.

### 4.2 **MPC Control Algorithm**:

*   Rolling Prediction: Based on the current state and prediction model, stepwise predicts future states and outputs.
*   Cost Function: Used to measure the difference between predicted output and actual output.
*   Optimization Algorithm: Used to find the control input that minimizes the cost function.

### **4.3 Optimization Algorithms**:

*   Non-negative constraints: Ensures that the control input is non-negative.
*   Boundary constraints: Ensures that the control input is within a specified range.


### **4.4 Usage Examples**:

*   Example code demonstrates how to use TensorDL-MPC for MPC control, including initializing the system, training the model, performing MPC control, simulating system dynamics, etc.



#### **4.4.1 . Usage Steps**

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




