from setuptools import setup, find_packages

setup(
    name="parallel_image_processing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.53.0",
        "pillow>=8.0.0",
        "scipy>=1.6.0",
        "matplotlib>=3.3.0",
        "cupy-cuda11>=10.0.0",  # Adjust based on your CUDA version
        "torch>=1.9.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.18.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A high-performance parallel image processing pipeline",
    keywords="image processing, parallel computing, GPU, SIMD",
    python_requires=">=3.8",
)
