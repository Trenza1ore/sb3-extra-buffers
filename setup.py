import os
from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

with open(os.path.join("sb3_extra_buffers", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

setup(
    name="sb3_extra_buffers",
    version=__version__,
    author="Hugo (Jin Huang)",
    author_email="SushiNinja123@outlook.com",
    url="https://github.com/Trenza1ore/sb3-extra-buffers",
    description="Extra buffer classes for Stable-Baselines3, reduce memory usage with minimal overhead.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "stable_baselines3",
        "tqdm"
    ],
    extras_require={
        "isal": ["isal"],
        "numba": ["numba"],
        "zstd": ["zstd"],
        "lz4": ["lz4"],
        "fast": ["sb3_extra_buffers[isal,numba,zstd,lz4]"],
        "extra": ["stable_baselines3[extra]"],
        "atari": ["sb3_extra_buffers[extra]"],
        "vizdoom": ["sb3_extra_buffers[extra]", "vizdoom"],
    },
    # PyPI package information.
    project_urls={
        "Source Code": "https://github.com/Trenza1ore/sb3-extra-buffers",
        "Stable-Baselines3": "https://github.com/DLR-RM/stable-baselines3",
        "Stable-Baselines3 - Contrib": "https://github.com/Stable-Baselines-Team/stable-baselines3-contrib",
    },
    package_data={
        "sb3_extra_buffers": ["version.txt"]
    }
)
