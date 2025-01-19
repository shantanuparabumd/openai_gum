from setuptools import setup, find_packages

setup(
    name="quadrobo_gym",
    version="0.1",
    packages=find_packages(),
    install_requires=["gym", "mujoco"],
    entry_points={
        "gym.envs": [
            "QuadroboEnv-v0 = quadrobo_gym.env:QuadroboEnv",
        ]
    },
    include_package_data=True,
    package_data={
        "quadrobo_gym": ["assets/*.xml", "assets/meshes/*.stl"],
    },
)
