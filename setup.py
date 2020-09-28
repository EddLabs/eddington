#!/usr/bin/env python
import os
import sys

from setuptools import setup

READTHEDOCS = "READTHEDOCS"

version = "0.0.17"
on_rtd = os.environ.get(READTHEDOCS) == "True"
install_requires = ["click >= 7.1.2", "ptable >= 0.7.2"]
if [sys.version_info.major, sys.version_info.minor] == [3, 6]:
    install_requires += ["dataclasses >= 0.7"]
if not on_rtd:
    install_requires.extend(
        [
            "scipy >= 1.5.2",
            "numpy >= 1.19.2",
            "openpyxl >= 3.0.5",
            "matplotlib >= 3.3.2",
        ]
    )
setup(version=version, install_requires=install_requires)
