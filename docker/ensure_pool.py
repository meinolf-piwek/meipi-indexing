"""Resolve or create a datapool id for the Docker entrypoint."""

from __future__ import annotations

import os
import sys

import sqlalchemy as sa

from meipi.indexing.model import DBPool
from meipi.indexing.operations import DBOperations


def main() -> int:
    name = os.environ.get("MEIPI_POOL_NAME")
    rootpath = os.environ.get("MEIPI_POOL_ROOTPATH")
    if not name or not rootpath:
        print("MEIPI_POOL_NAME and MEIPI_POOL_ROOTPATH are required", file=sys.stderr)
        return 1

    dbop = DBOperations(allow_no_pool=True)
    with dbop.Session() as session:
        pool = session.scalars(sa.select(DBPool).where(DBPool.pool == name)).first()
        if pool is None:
            pool = DBPool(pool=name, rootpath=rootpath)
            session.add(pool)
            session.commit()
            session.refresh(pool)
            print(f"Created datapool id={pool.id} name={pool.pool!r}", file=sys.stderr)
        elif pool.rootpath != rootpath:
            print(
                f"Datapool {name!r} exists with rootpath {pool.rootpath!r}, "
                f"expected {rootpath!r}",
                file=sys.stderr,
            )
            return 1
    print(pool.id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
