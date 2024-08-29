from setuptools import setup, find_packages

setup(
    name="sales_forecasting",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "mlflow",
        "Flask",
    ],
    entry_points={
        "console_scripts": [
            "train=sales_forecasting.train:main",
            "inference=sales_forecasting.inference:main",
            "prepare_data=sales_forecasting.data_preparation:main",
        ],
    },
)