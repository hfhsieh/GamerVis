import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="GamerVis",
    version="0.1",
    author="He-Feng Hsieh",
    author_email="x.geometric@gmail.com",
    description=("Python modules for visualizing GAMER data for CCSN simulations."),
    license="BSD",
    url="https://github.com/hfhsieh/GamerVis",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy<2.2,>=1.24", "scipy", "matplotlib>=3.2", "yt", "pycwt"],
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research"
    ],
)
