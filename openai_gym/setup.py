from setuptools import setup, find_packages

setup(
    name="openai_gym_rl",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "hydra-core",
        "torch",
        "numpy"
    ],
)
