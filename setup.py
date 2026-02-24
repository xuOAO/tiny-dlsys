from setuptools import setup, find_packages

setup(
    name="tiny-dlsys",
    version="0.1.0",
    description="一个面向教学的微型深度学习系统，支持单机单卡（CPU / CUDA）",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
    extras_require={
        "cuda": ["cupy-cuda12x", "triton"],   # 按实际 CUDA 版本选择 cupy-cudaXXX
    },
)
