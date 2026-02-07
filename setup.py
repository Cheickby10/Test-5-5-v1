from setuptools import setup, find_packages

setup(
    name="fifa_analytics_bot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        # Liste identique à requirements.txt
        "streamlit==1.28.1",
        "pandas==2.1.4",
        "numpy==1.24.3",
        "scikit-learn==1.3.2",
        # ... autres dépendances
    ],
)
