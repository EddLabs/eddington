#!/usr/bin/env python
import os

from setuptools import setup

version = "0.0.24.dev1"
install_requires = [
    "click >= 8.1.3",
    "prettytable >= 3.4.0",
    'scipy >= 1.9.1; python_version > "3.7"',
    'scipy >= 1.7.2, <1.8; python_version == "3.7"',
    "numpy >= 1.23.2",
    "openpyxl >= 3.0.10",
    "matplotlib >= 3.5.3",
    "types-mock >= 4.0.15",
]

if os.environ.get("READTHEDOCS") == "True":
    install_requires = install_requires[:3]

setup(version=version, install_requires=install_requires)
