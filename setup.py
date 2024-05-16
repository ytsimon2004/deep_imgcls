#!/usr/bin/env python
# Install script for analysis code
import sys

sys.path.insert(0, 'src')


def get_dependencies() -> list[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        requires = []
        for line in f:
            req = line.split("#", 1)[0].strip()
            if req and not req.startswith("--"):
                requires.append(req)
    return requires



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
        install_requires=get_dependencies()
    )
