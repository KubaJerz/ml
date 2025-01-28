# setup.py
from  setuptools import find_packages, setup
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name = "ml_framework",
    version = "1.0.3",
    author = "Kuba Jerzmanowski",
    description = "A modular machine learning framework",
    long_description=long_description, 
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires = [
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "matplotlib_inline>=0.1.7",
        "tqdm>=4.65.0",
        "scikit-learn>=1.0.0",
        "torcheval>=0.0.7",
        "pyyaml>=6.0.2",
        "IPython>=8.31.0"]
    )