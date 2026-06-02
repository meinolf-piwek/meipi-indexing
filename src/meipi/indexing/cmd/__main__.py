"""Console entry: apply ``--env-file`` before ``meipi.indexing`` is imported."""

from __future__ import annotations

import os
import sys


def _apply_env_file_from_argv(argv: list[str]) -> None:
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--env-file":
            if i + 1 < len(argv):
                os.environ["MEIPI_CONFIG_ENV"] = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--env-file="):
            os.environ["MEIPI_CONFIG_ENV"] = arg.partition("=")[2]
        i += 1


def main() -> None:
    _apply_env_file_from_argv(sys.argv[1:])
    from meipi.indexing.cmd.cli import main as cli_main

    raise SystemExit(cli_main())
