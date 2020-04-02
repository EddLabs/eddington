#!/usr/bin/env python
from pathlib import Path

from setuptools import setup
import io
import re

with io.open(
    Path("src") / "eddington_core" / "__init__.py", encoding="utf8"
) as version_file:
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file.read(), re.M
    )
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

setup(version=version)
