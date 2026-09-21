# SPDX-License-Identifier: MIT
"""`python -m calibrix` -> the solo runner (whole platform, one command)."""
import sys

from .solo import main

if __name__ == "__main__":
    sys.exit(main())
