"""Entry point for ``python -m traffic_insights``.

This module only wires together the CLI, configuration loader and
pipeline factory. All business logic lives in the pipeline packages.
"""

from __future__ import annotations

import sys


def main():
    """Initial Greetings!"""
    print("Hello World!")


if __name__ == "__main__":
    sys.exit(main())
