"""CLI for meipi-indexing."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import click

from .. import appconf
from ..model import DBPool
from ..operations import DBOperations
from .main import read_files_bulk


def _db_ops(pool_id: int) -> DBOperations:
    return DBOperations(pool_id=pool_id)


@click.group()
@click.option(
    "--env-file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help=(
        "Path to config.env; must be passed on the command line before import "
        "(handled by the meipi-index entry point). Restart the process to apply changes."
    ),
)
@click.pass_context
def cli(ctx: click.Context, env_file: str | None) -> None:
    """Index documents and images into PostgreSQL."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = appconf


@cli.command("schema-info")
@click.option("--json", "as_json", is_flag=True, help="Print JSON output")
@click.pass_context
def schema_info(ctx: click.Context, as_json: bool) -> None:
    """Show database schema and tables."""
    dbop = DBOperations(allow_no_pool=True)
    info = dbop.schema_info()
    if as_json:
        click.echo(json.dumps(info, indent=2))
    else:
        click.echo(f"Schema: {info['schema']}",color=True)
        click.echo("Tables:",color=True)
        for name, count in info["tables"].items():
            click.echo(f"  {name:<20}: {count:>10,} rows",color=True)


@cli.command("create-tables")
@click.option(
    "--recreate",
    is_flag=True,
    help="Drop and recreate all tables (destructive)",
)
@click.pass_context
def create_tables(ctx: click.Context, recreate: bool) -> None:
    """Create database tables."""
    dbop = DBOperations(allow_no_pool=True)
    if recreate:
        dbop.recreate_tables()
        click.echo("Tables recreated.")
    else:
        dbop.create_tables()
        click.echo("Tables created (existing tables unchanged).")


@cli.command("create-pool")
@click.option("--name", required=True, help="Unique pool name")
@click.option(
    "--rootpath",
    required=True,
    help="Filesystem root for files in this pool",
)
@click.option("--description", default=None, help="Optional description")
@click.pass_context
def create_pool(
    ctx: click.Context,
    name: str,
    rootpath: str,
    description: str | None,
) -> None:
    """Register a new datapool."""
    dbop = DBOperations(allow_no_pool=True)
    pool = DBPool(pool=name, rootpath=rootpath, description=description)
    created = dbop.create_pool(pool)
    click.echo(
        f"Created pool id={created.id} name={created.pool!r} rootpath={created.rootpath!r}"
    )


@cli.command("clear-pool")
@click.option(
    "--pool-id",
    type=int,
    required=True,
    help="Datapool id from the datapools table",
)
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def clear_pool(ctx: click.Context, pool_id: int, yes: bool) -> None:
    """Delete all filemeta for a pool."""
    dbop = _db_ops(pool_id)
    if not yes and not click.confirm(
        f"Delete all filemeta rows for pool {pool_id}?"
    ):
        click.echo("Aborted.")
        return
    dbop.clear_pool()
    click.echo(f"Cleared pool {pool_id}.")


@cli.command("read-files")
@click.option(
    "--pool-id",
    type=int,
    required=True,
    help="Datapool id from the datapools table",
)
@click.argument("relpath")
@click.option(
    "--batch-size",
    type=int,
    default=500,
    show_default=True,
    help="Files per batch",
)
@click.option(
    "--no-incremental",
    is_flag=True,
    help="Re-read files even if already in the database",
)
@click.option(
    "--no-thumbs",
    is_flag=True,
    help="Skip thumbnail generation after ingest",
)
@click.pass_context
def read_files(
    ctx: click.Context,
    pool_id: int,
    relpath: str,
    batch_size: int,
    no_incremental: bool,
    no_thumbs: bool,
) -> None:
    """Ingest files from a pool directory into the database."""
    asyncio.run(
        read_files_bulk(
            pool_id=pool_id,
            relpath=relpath,
            batchsize=batch_size,
            incremental=not no_incremental,
            update_thumbs=not no_thumbs,
            config=appconf,
        )
    )


@cli.command("update-thumbs")
@click.option(
    "--pool-id",
    type=int,
    required=True,
    help="Datapool id from the datapools table",
)
@click.option(
    "--heic-only",
    is_flag=True,
    help="Only run the DALI pass (non-HEIC files)",
)
@click.option(
    "--remainder-only",
    is_flag=True,
    help="Only run the PIL fallback pass",
)
@click.pass_context
def update_thumbs(
    ctx: click.Context,
    pool_id: int,
    heic_only: bool,
    remainder_only: bool,
) -> None:
    """Generate missing thumbnails for pictures in a pool."""
    if heic_only and remainder_only:
        raise click.UsageError("--heic-only and --remainder-only are mutually exclusive")

    dbop = _db_ops(pool_id)
    failed: list[int] = []
    if heic_only:
        failed = dbop.update_thumbs_no_heic()
    elif remainder_only:
        failed = dbop.update_thumbs_no_thumb()
    else:
        failed.extend(dbop.update_thumbs_no_heic())
        failed.extend(dbop.update_thumbs_no_thumb())
    if failed:
        click.echo(f"Failed picture ids: {failed}")
    else:
        click.echo("All thumbnails updated successfully.")


@cli.command("insert-pics")
@click.option(
    "--pool-id",
    type=int,
    required=True,
    help="Datapool id from the datapools table",
)
@click.pass_context
def insert_pics(ctx: click.Context, pool_id: int) -> None:
    """Create picture rows for indexed image filemeta."""
    dbop = _db_ops(pool_id)
    dbop.insert_pics_from_meta()
    click.echo(f"Inserted picture rows for pool {pool_id}.")


@cli.command("insert-docs")
@click.option(
    "--pool-id",
    type=int,
    required=True,
    help="Datapool id from the datapools table",
)
@click.option(
    "--ocr",
    is_flag=True,
    help="Use the OCR Tika endpoint instead of the no-OCR endpoint",
)
@click.pass_context
def insert_docs(ctx: click.Context, pool_id: int, ocr: bool) -> None:
    """Extract and store document text for indexed filemeta."""
    dbop = _db_ops(pool_id)

    async def _run() -> None:
        await dbop.insert_docs_from_meta(skipocr=not ocr)

    asyncio.run(_run())
    click.echo(f"Inserted document rows for pool {pool_id}.")


def main(args: list[str] | None = None) -> int:
    """Entry point for the ``meipi-index`` console script."""
    #warnings.filterwarnings("ignore")
    try:
        cli.main(args=args, prog_name="meipi-index", standalone_mode=False)
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except KeyboardInterrupt:
        click.echo("Interrupted.", err=True)
        return 130
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
