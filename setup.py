from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="parallel-opt",
    version="1.0.0",
    author="CMKL Team",
    author_email="info@example.com",
    description="Parallelism Optimization Framework for Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/parallelism-optimization-framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "networkx>=2.6.3",
        "psutil>=5.9.0",
        "pynvml>=11.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "plotly>=5.5.0",
        "pyyaml>=6.0.0",
        "pandas>=1.3.5",
    ],
    extras_require={
        "full": [
            "scikit-learn>=1.0.0",
            "metis>=0.2a5",
            "ray>=1.13.0",
        ],
    },
)
