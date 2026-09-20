from setuptools import setup, find_packages
import glob
import os

setup(
    name="wan-animate-2",
    version="0.0.1",
    author="Guangyuan Wang",
    author_email="2207673890@qq.com",
    description="wan-animate-2 inference Python Package.",
    license="Apache-2.0",
    packages=find_packages(exclude=('examples', 'examples.*', 'tests', 'tests.*', "tools")),
)