# Installation Guide

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
pip install scikit-learn  #pip install scikit-learn == 1.2.2
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

# Method 3: Install 

### **1. Activate Anaconda Virtual Environment**

If you have not already activated an Anaconda virtual environment, please follow these steps:

- Open Command Prompt (CMD).
- Enter the following command to create a new virtual environment (e.g., `TensorDL-MPC_env`) and activate it:

```bash
conda create --name TensorDL-MPC_env python=3.9
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
python setup.py easy_install
```

### **4. Verify Installation**

1. Enter the following command to verify that the `setup.py` has installed the project:

```bash
python -c "import tensordlmpc"
```

If the command does not produce any errors, it indicates that the project has been installed successfully.

### **5. Run Examples**

1. Enter the following command to run an example script from the project (if available):

```bash
python example.py
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
