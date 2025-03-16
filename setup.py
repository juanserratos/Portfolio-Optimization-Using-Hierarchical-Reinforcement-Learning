"""
Setup script for the deep_rl_trading package.
"""
from setuptools import setup, find_packages

setup(
    name="deep_rl_trading",
    version="0.1.0",
    description="Deep Reinforcement Learning Framework for Adaptive Trading Strategies",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/username/deep-rl-trading",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "torch>=1.10.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "gym>=0.21.0",
        "yfinance>=0.1.74",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
)