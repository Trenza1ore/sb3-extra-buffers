__all__ = ["package_version"]

import os

with open(os.path.join(os.path.dirname(__file__), "version.txt")) as f:
    package_version = f.read().strip()
