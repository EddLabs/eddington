#!/usr/bin/env python
import os

from setuptools import setup

version = "0.0.24.dev1"
install_requires = [
    "click >= 7.1.2",
    "prettytable >= 2.0.0",
    'dataclasses >= 0.8; python_version == "3.6"',
    'scipy >= 1.6.0; python_version > "3.6"',
    'scipy >= 1.5.4, < 1.6; python_version == "3.6"',
    "numpy >= 1.19.5",
    "openpyxl >= 3.0.6",
    "matplotlib >= 3.3.3",
    "types-mock >= 4.0.4",
]

if os.environ.get("READTHEDOCS") == "True":
    install_requires = install_requires[:3]

setup(version=version, install_requires=install_requires)
