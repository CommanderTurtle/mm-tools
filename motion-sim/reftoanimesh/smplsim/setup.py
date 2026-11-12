# Copyright (c) Meta Platforms, Inc. and affiliates.
#


from setuptools import setup, find_packages

setup(
    name='smpl_sim',
    version='0.1',
    packages=find_packages(),
    package_data={
        # Include any *.txt files found in the 'your_package' package
        'smpl_sim': ['*.xml', '*.urdf', "*.yaml" ],
    },
    # Other setup parameters...
)