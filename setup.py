from setuptools import setup, find_packages

setup(
    name="dlmpc",
    version="1.0",
    author="Xiaoming Wang",
    author_email="wangxiaoming19951@163.com",
    description="A toolbox for deep learning-based nonlinear model predictive control",
    packages=find_packages(),
    install_requires=[
        "tensorflow-gpu==2.9.0",
        "scikit-learn==1.2.2",
        "pandas==1.3.5",
        "openpyxl",
        "matplotlib"
    ]
)
