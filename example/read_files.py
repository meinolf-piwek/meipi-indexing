"""Example: ingest files via CLI (preferred) or programmatically."""

import os
import subprocess
import sys

os.environ.setdefault("IND_PG_SCHEMA", "public")
os.environ.setdefault("IND_DATA_DIR", "/home/padmin/Development/projekte/meipi-indexing/data")

# CLI usage:
#   meipi-index read-files --pool-id 2 "Bilder 2018"

if __name__ == "__main__":
    sys.exit(
        subprocess.run(
            [
                "meipi-index",
                "read-files",
                "--pool-id",
                "2",
                "Bilder 2018",
            ],
            check=False,
        ).returncode
    )
