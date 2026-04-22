from setuptools import setup, find_packages

setup(
    name="plastic-transfer",
    version="0.1.0",
    description="Plastic Transfer: skill-based transfer learning for RL",
    author="Jorge Hernandez",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "stable-baselines3",
        "gymnasium"
    ],
    python_requires=">=3.8",
)