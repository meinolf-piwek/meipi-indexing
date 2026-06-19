"""Bulk ingest workflows used by the CLI."""

from __future__ import annotations

import asyncio
from itertools import batched

import sqlalchemy as sa
from tqdm.auto import tqdm

from ..config import Config, FTYPE, resolve_config
from ..model import DBMeta, DBPool
from ..operations import AsyncFileOperations, DBOperations


async def index_file(
    *,
    pool_id: int | None = None,
    pool: DBPool | None = None,
    rel_path: str,
    update_thumbs: bool = True,
    store_chunks: bool = True,
    config: Config | None = None,
) -> bool:
    """Index or re-index a single file in the database."""
    config = resolve_config(config)
    if pool_id is not None:
        dbop = DBOperations(pool_id=pool_id, config=config)
        pool = dbop.pool
    elif pool is not None:
        dbop = DBOperations(pool=pool, config=config)
    else:
        raise ValueError("Either pool_id or pool must be provided")

    async with AsyncFileOperations(pool, config) as afop:
        dbmeta = await afop.file_to_db(rel_path)
        if dbmeta is None:
            return False

        with dbop.Session() as session:
            session.execute(
                sa.delete(DBMeta).where(
                    DBMeta.pool_id == pool.id,
                    DBMeta.path == rel_path,
                )
            )
            session.flush()
            session.add(dbmeta)
            session.flush()
            session.commit()

    if store_chunks and dbmeta.inhalt.strip():
        dbop.upsert_document_chunks(dbmeta.id, dbmeta.inhalt)

    if update_thumbs:
        if dbmeta.ftype == FTYPE.DOC:
            dbop.update_thumb_for_doc(dbmeta.id, rel_path)
        elif dbmeta.ftype == FTYPE.PIC and dbmeta.pic is not None:
            dbop.update_thumb_for_pic(dbmeta.pic.id, rel_path)
    return True


async def read_files_bulk(
    *,
    pool_id: int | None = None,
    pool: DBPool | None = None,
    relpath: str,
    batchsize: int = 500,
    incremental: bool = True,
    update_thumbs: bool = True,
    store_chunks: bool = True,
    config: Config | None = None,
) -> None:
    """Read files from the filesystem, extract metadata, and store them in the DB.

    Args:
        pool_id: Id of an existing datapool row (preferred for CLI usage).
        pool: In-memory pool object (e.g. for scripts/notebooks).
        relpath: Path relative to ``IND_DOCROOT``.
        batchsize: Number of files processed per batch.
        incremental: Skip files whose path is already in filemeta for this pool.
        update_thumbs: Run thumbnail generation after ingest.
        store_chunks: Chunk document text into ``bge_m3_vectors`` after ingest.
        config: Application configuration (defaults to ``appconf``).
    """
    config = resolve_config(config)
    if pool_id is not None:
        dbop = DBOperations(pool_id=pool_id, config=config)
        pool = dbop.pool
    elif pool is not None:
        dbop = DBOperations(pool=pool, config=config)
    else:
        raise ValueError("Either pool_id or pool must be provided")

    async with AsyncFileOperations(pool, config) as afop:
        files = list(afop.dir_tree(relpath))
        if incremental:
            with dbop.Session() as session:
                existing = session.scalars(
                    sa.select(DBMeta.path).where(DBMeta.pool_id == pool.id)
                ).all()
                existing_set = set(existing)
            files = [path for path in files if path not in existing_set]

        batches = list(batched(files, batchsize))
        tika_sem = asyncio.Semaphore(min(20, batchsize))
        for batch in tqdm(
            batches,
            desc=f"Reading {len(files)} files in {len(batches)} batches",
            total=len(batches),
            unit="batch",
        ):
            async def _index_one(file: str) -> DBMeta | None:
                async with tika_sem:
                    return await afop.file_to_db(file)

            tasks = [asyncio.create_task(_index_one(file)) for file in batch]
            metalist = []
            async for task in asyncio.as_completed(tasks):
                dbmeta = await task
                if dbmeta is not None:
                    metalist.append(dbmeta)
            if metalist:
                with dbop.Session() as session:
                    session.add_all(metalist)
                    session.flush()
                    session.commit()
                if store_chunks:
                    for meta in metalist:
                        if meta.inhalt.strip():
                            dbop.upsert_document_chunks(meta.id, meta.inhalt)

    if update_thumbs:
        print("Updating thumbs")
        dbop.update_thumbs_no_heic()
        dbop.update_thumbs_no_thumb()
        dbop.update_thumbs_doc()
        dbop.update_phashes()
