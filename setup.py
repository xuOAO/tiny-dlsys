from setuptools import setup, find_packages

setup(
    name="tiny-dlsys",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    extras_require={
        "cuda": ["cupy-cuda12x", "triton"],   # 按实际 CUDA 版本选择 cupy-cudaXXX
    },
    python_requires=">=3.8",
)
