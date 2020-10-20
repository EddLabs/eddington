#!/usr/bin/env python
import os

from setuptools import setup


version = "0.0.21"
install_requires = [
    "click >= 7.1.2",
    "prettytable >= 1.0.1",
    'dataclasses >= 0.7; python_version == "3.6"',
    "scipy >= 1.5.3",
    "numpy >= 1.19.2",
    "openpyxl >= 3.0.5",
    "matplotlib >= 3.3.2",
]

if os.environ.get("READTHEDOCS") == "True":
    install_requires = install_requires[:3]

setup(version=version, install_requires=install_requires)
