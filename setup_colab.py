#!/usr/bin/env python
# Install script for imgcls module
import sys

sys.path.insert(0, 'src')

packages = [
    'imgcls.classification',
]

if __name__ == '__main__':
    from setuptools import setup

    #
    setup(
        name='imgcls',
        version='0.0.0',
        author='Yu-Ting',
        author_email='ytsimon2004@gmail.com',
        description='code for simply image classification',
        license='GPL',
        packages=packages,
        package_dir={'': 'src'},
    )
