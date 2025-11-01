"""Small wrapper so you can run Streamlit with a short poetry script.

Usage:
  poetry run app [-- <streamlit-args>]

This will execute: `streamlit run app/main.py <streamlit-args>`
"""
from __future__ import annotations

import subprocess
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run Streamlit pointing to app/main.py, forwarding any args.

    This uses the `streamlit` executable from the current environment, so
    running `poetry run app` will execute the project's Streamlit install.
    """
    if argv is None:
        argv = sys.argv[1:]

    cmd = ["streamlit", "run", "app/main.py"] + list(argv)

    # Use subprocess.call so exit code is returned to the caller.
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        # streamlit not installed in environment
        print(
            "Error: 'streamlit' executable not found in the environment. \n"
            "Install streamlit in your Poetry env or run via `poetry run streamlit ...`.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
