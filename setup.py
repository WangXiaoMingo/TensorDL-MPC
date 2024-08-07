from setuptools import setup, find_packages

setup(
    name="dlmpc",
    version="1.0",
    author="Xiaoming Wang",
    author_email="wangxiaoming19951@163.com",
    description="A toolbox for deep learning-based nonlinear model predictive control",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "numpy",
        "matplotlib",
        "pandas",
        "scikit-learn"
    ]
)
