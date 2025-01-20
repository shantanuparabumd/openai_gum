from setuptools import setup, find_packages

setup(
    name="openai_gym_rl",
    version="0.1",
    author="Shantanu Parab",
    author_email="sparab@umd.edu",
    description="Reinforcement Learning using OpenAI Gym and MuJoCo",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shantanuparabumd/openai_gym_rl",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "hydra-core",
        "torch",
        "numpy",
        "matplotlib",
        "omegaconf",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    include_package_data=True,
)
