"""Entry point for `python -m fant2` — dispatches to fant2.cli.main."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
