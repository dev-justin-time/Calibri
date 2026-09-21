# SPDX-License-Identifier: MIT
"""`python -m calibrix.agents` -> run the multi-agent pipeline."""
import sys

from .orchestrator import main

if __name__ == "__main__":
    sys.exit(main())
