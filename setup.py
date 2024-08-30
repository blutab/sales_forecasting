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
            "train=app.train:main",
            "inference=app.inference:main",
            "prepare_data=app.data_preparation:main",
        ],
    },
)