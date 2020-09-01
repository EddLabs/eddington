#!/usr/bin/env python
import os

from setuptools import setup

READTHEDOCS = "READTHEDOCS"

version = "0.0.15"
on_rtd = os.environ.get(READTHEDOCS) == "True"
install_requires = []
if not on_rtd:
    install_requires = ["scipy >= 1.5.2", "numpy >= 1.19.1", "openpyxl >= 3.0.4"]
setup(version=version, install_requires=install_requires)
