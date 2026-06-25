"""Enable ``python -m genai_studio ...`` (the package's CLI entry point)."""

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())
