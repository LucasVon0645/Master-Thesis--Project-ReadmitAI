"""Wrapper to start the FastAPI app via Uvicorn using a short poetry script.

Usage:
  poetry run api [-- <uvicorn-args>]

This will execute: `uvicorn api.main:app <uvicorn-args>`
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Sequence


def _should_reload(argv: Sequence[str]) -> bool:
    """Decide whether to enable uvicorn --reload.

    Rules:
    - If the user explicitly passed --reload, respect that (don't add again).
    - If env var API_RELOAD is set to a falsey value (0, false, no), disable reload.
    - Otherwise enable reload by default (helpful during development).
    """
    # If user explicitly passes --reload, don't add it again
    for a in argv:
        if a == "--reload" or a.startswith("--reload"):
            return False

    env = os.getenv("API_RELOAD", "1")
    if str(env).lower() in ("0", "false", "no", "off"):
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Run uvicorn for `api.main:app`, forwarding CLI args and enabling reload by default.

    Usage examples:
      poetry run api                # starts uvicorn with --reload by default
      API_RELOAD=0 poetry run api    # starts without --reload
      poetry run api -- --host 0.0.0.0 --port 8080 --reload  # explicit reload
    """
    if argv is None:
        argv = sys.argv[1:]

    # Base app target
    app_target = "api.main:app"

    # Default args to use when user passed no args
    default_args = [app_target, "--host", "127.0.0.1", "--port", "8000"]

    if len(argv) == 0:
        # Add --reload if allowed
        if _should_reload(argv):
            default_args.append("--reload")
        cmd = ["uvicorn"] + default_args
    else:
        # Use provided args; append --reload if not present and allowed
        args = list(argv)
        if _should_reload(args):
            args = args + ["--reload"]
        cmd = ["uvicorn", app_target] + args

    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print(
            "Error: 'uvicorn' executable not found in the environment.\n"
            "Install uvicorn in your Poetry env or run via `poetry run uvicorn ...`.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
