from __future__ import annotations

"""Database and filesystem operations for indexing workflows.

``DBOperations`` wraps SQLAlchemy session/engine handling and higher-level
update routines. ``AsyncFileOperations`` handles file discovery plus Tika/XMP
extraction for ``DBMeta`` and child ORM rows.
"""
import io
import os
from typing import List, Optional, Tuple, Generator, Sequence, Any, cast
from datetime import datetime
import json
from pathlib import Path
from tika_client import AsyncTikaClient, TikaKey
from tika_client.data_models import TikaResponse
#import asyncio
#from itertools import batched
#from tqdm.auto import tqdm
from hashlib import file_digest
from PIL import Image
#import tika.parser as tp
import numpy as np
from libxmp import utils as xmpu
import sqlalchemy as sa
from sqlalchemy import Table
from sqlalchemy.orm import sessionmaker, Session
from .model import Base as ModelBase, DBMeta, DBPic, DBDinoV2Vector, DBBgeM3Vector, DBVid, IdList, DBPool
from .config import Config, FTYPE, resolve_config
from .text_cleaning import clean_document_text
from .embedding.text_preprocess import chunks_to_rows

WATCHER_TABLES: tuple[type[ModelBase], ...] = (
    DBPool,
    DBMeta,
    DBPic,
    
    DBVid,
)

def _dali_resizer(*args, **kwargs):
    from .dali import DALIImageResizer

    return DALIImageResizer(*args, **kwargs)


class DBOperations():
    """High-level PostgreSQL operations for one configured data pool."""
    def __init__(
        self,
        pool_id: int | None = None,
        pool: DBPool | None = None,
        *,
        allow_no_pool: bool = False,
        config: Config | None = None,
        enginekwargs: dict | None = None,
        sessionkwargs: dict | None = None,
    ):
        config = resolve_config(config)
        self.config = config
        enginekwargs = {} if not enginekwargs else enginekwargs
        sessionkwargs = {} if not sessionkwargs else sessionkwargs
        self.metadata = config.metadata
        self.pg_schema = config.pg_schema
        connect_args = {"options": f"-c search_path={config.pg_schema}"}
        self.engine = sa.create_engine(self.config.db_conn_URL, connect_args = connect_args,
        logging_name="DBOperations",pool_logging_name=self.config.logger_name, **enginekwargs)
        self.logger = self.config.logger
        try:
            self.Session = sessionmaker(
                bind=self.engine, expire_on_commit=False, **sessionkwargs
            )
        except Exception as e:
            self.logger.error("Error creating PostgreSQL engine: %s", e)
            raise
        else:
            self.logger.info("PostgreSQL engine created successfully.")
        self._document_chunker_instance: Any = None
        if pool_id is not None:
            self.get_pool(pool_id)
        elif pool is not None:
            #self.create_pool(pool)
            self.pool = pool
        elif allow_no_pool:
            self.logger.warning(
                "No pool provided; filesystem operations require pool_id or pool"
            )
            self.pool = DBPool(
                id=0, pool="default", description="Default pool"
            )
        else:
            raise ValueError("Either pool_id or pool must be provided")
        self.docroot = config.resolved_docroot()

    def schema_info(self) -> dict[str, Any]:
        """Get the schema information of the database."""
        tables = {}
        with self.Session() as session:
            cat = sa.Table('pg_tables', sa.MetaData(schema='pg_catalog'), autoload_with=self.engine)
            for tname in session.scalars(
                sa.select(cat.c.tablename).where(cat.c.schemaname == self.pg_schema)
            ):
                table = sa.Table(tname, self.metadata, autoload_with=self.engine)
                stmt = sa.select(sa.func.count()).select_from(table)
                tables[tname] = session.scalars(stmt).one()
        return {"schema": self.pg_schema, "tables": tables} 
        
    def get_pool(self, pool_id: int)-> DBPool:
        """Get the pool with the given id."""
        with self.Session() as session:
            self.pool = session.get(DBPool, pool_id)
            if self.pool is None:
                raise ValueError(f"Pool with id {pool_id} not found")
            return self.pool

    def create_pool(self, pool: DBPool):
        """Insert a new data pool and make it active for this instance."""
        with self.Session() as session:
            session.add(pool)  
            session.commit()
            session.refresh(pool)
            self.pool = pool
            self.logger.info("Pool created with id %s", pool.id)
            return pool
    
    def create_tables(self, entities: Optional[Sequence[type[ModelBase]]] = None):
        """Erstellt die Tabellen in der Datenbank, falls sie noch nicht existieren."""
        if entities:
            tables = cast(list[Table], [entity.__table__ for entity in entities])
        else:
            tables = None
        self.metadata.create_all(self.engine, tables=tables)

    def recreate_tables(self, entities: Optional[Sequence[type[ModelBase]]] = None):
        """Recreate tables in the database."""
        if entities:
            tables = cast(list[Table], [entity.__table__ for entity in entities])
        else:
            tables = None
        self.metadata.drop_all(self.engine, tables=tables)
        self.metadata.create_all(self.engine, tables=tables)

    def ensure_tables(
        self, entities: Sequence[type[ModelBase]] | None = None
    ) -> list[str]:
        """Create ORM tables that are missing in the database."""
        entities = tuple(entities or WATCHER_TABLES)
        existing = set(self.schema_info()["tables"])
        missing = [entity for entity in entities if entity.__tablename__ not in existing]
        if missing:
            self.create_tables(entities=missing)
        return [entity.__tablename__ for entity in missing]

    def count_pool_filemeta(self) -> int:
        """Return the number of filemeta rows for the current pool."""
        with self.Session() as session:
            return int(
                session.scalar(
                    sa.select(sa.func.count())
                    .select_from(DBMeta)
                    .where(DBMeta.pool_id == self.pool.id)
                )
                or 0
            )

    def list_pool_file_paths(self, watch_relpath: str = ".") -> set[str]:
        """Return indexed paths for the current pool inside the watched subtree."""
        from .paths import normalize_rel_path

        watch_relpath = normalize_rel_path(watch_relpath)
        stmt = sa.select(DBMeta.path).where(DBMeta.pool_id == self.pool.id)
        if watch_relpath not in ("", "."):
            prefix = watch_relpath + "/"
            stmt = stmt.where(
                sa.or_(DBMeta.path == watch_relpath, DBMeta.path.like(f"{prefix}%"))
            )
        with self.Session() as session:
            paths = session.scalars(stmt).all()
        return {normalize_rel_path(path) for path in paths}
        
         
    def clear_pool(self):
        """Löscht alle Daten aus dem angegebenen Pool."""
        with self.Session() as session:
            session.query(DBMeta).filter(DBMeta.pool_id == self.pool.id).delete()
            session.flush()
            session.commit()

    def delete_file_meta(self, rel_path: str) -> bool:
        """Remove the filemeta row for ``rel_path`` in the current pool."""
        with self.Session() as session:
            deleted = (
                session.query(DBMeta)
                .filter(DBMeta.pool_id == self.pool.id, DBMeta.path == rel_path)
                .delete()
            )
            session.commit()
            if deleted:
                self.logger.info("Removed index entry for %s", rel_path)
            return deleted > 0

    def _document_chunker(self) -> Any:
        if self._document_chunker_instance is None:
            from .embedding.text_preprocess import ChunkConfig, DocumentChunker

            self._document_chunker_instance = DocumentChunker(
                ChunkConfig(clean_text=False)
            )
        return self._document_chunker_instance

    def insert_document_chunks(
        self, doc_id: int, contents: Sequence[str]
    ) -> int:
        """Insert chunk rows without removing existing ones."""
        with self.Session() as session:
            rows = chunks_to_rows(doc_id, contents)
            if rows:
                session.add_all(rows)
            session.commit()
            return len(rows)

    def replace_document_chunks(
        self, doc_id: int, contents: Sequence[str]
    ) -> int:
        """Replace all BGE-M3 chunk rows for one document."""
        with self.Session() as session:
            session.execute(
                sa.delete(DBBgeM3Vector).where(DBBgeM3Vector.doc_id == doc_id)
            )
            rows = chunks_to_rows(doc_id, contents)
            if rows:
                session.add_all(rows)
            session.commit()
            return len(rows)

    def chunk_and_store_document(
        self, meta: DBMeta, chunker: Any, *, doc_id: int | None = None
    ) -> int:
        """Chunk ``meta.inhalt`` and persist rows in ``bge_m3_vectors``."""
        doc_id = doc_id or meta.id
        contents = chunker.chunk_text(meta.inhalt)
        return self.replace_document_chunks(doc_id, contents)

    def store_chunks_for_meta(self, meta: DBMeta) -> int:
        """Chunk and persist rows for one indexed document filemeta."""
        
        if meta.inhalt is None or meta.inhalt.strip() == "":
            self.logger.warning(
                "Skipping chunks for %s: no inhalt", meta.path
            )
            return 0        
        return self.chunk_and_store_document(meta, self._document_chunker())

    def insert_chunks_for_meta(self, meta: DBMeta) -> int:
        """Chunk and insert rows for one document without touching existing chunks."""
        if meta.inhalt is None or meta.inhalt.strip() == "":
            self.logger.warning(
                "Skipping chunks for %s: no inhalt", meta.path
            )
            
        
        chunks = self._document_chunker().chunk_text(meta.inhalt)
        return self.insert_document_chunks(meta.id, chunks)

    def store_chunks_for_meta_list(self, metalist: Sequence[DBMeta]) -> int:
        """Chunk and persist rows for a batch of newly ingested filemeta rows."""
        total = 0
        for meta in metalist:
            try:
                total += self.store_chunks_for_meta(meta)
            except Exception:
                self.logger.exception("Failed to store chunks for %s", meta.path)
        return total

    def insert_missing_chunks_from_meta(self) -> int:
        """Chunk documents that have no rows in ``bge_m3_vectors`` yet."""
        from sqlalchemy.orm import joinedload

        has_chunks = (
            sa.select(DBBgeM3Vector.chunk_id)
            .select_from(DBMeta)
            .join(DBBgeM3Vector, DBBgeM3Vector.doc_id == DBMeta.id)
            .where(DBMeta.id == DBMeta.id)
            .correlate(DBMeta)
            .exists()
        )
        total = 0
        with self.Session() as session:
            stmt = (
                sa.select(DBMeta)
                .where(
                    DBMeta.pool_id == self.pool.id,
                    
                    DBMeta.inhalt != "",
                    ~has_chunks,
                )
            )
            metalist = session.execute(stmt).scalars().unique().all()
            session.flush()
            for meta in metalist:
                try:
                    total += self.insert_chunks_for_meta(meta)
                except Exception:
                    self.logger.exception(
                        "Failed to insert chunks for %s", meta.path
                    )
        return total

    def update_chunks_from_meta(self) -> int:
        """Re-chunk and replace rows for all doc filemeta in the pool."""
        from sqlalchemy.orm import joinedload

        total = 0
        with self.Session() as session:
            stmt = (
                sa.select(DBMeta)
                
                .where(
                    DBMeta.pool_id == self.pool.id,
                    
                    DBMeta.inhalt != "",
                )
            )
            metalist = session.execute(stmt).scalars().unique().all()
            
            session.flush()
            for meta in metalist:
                try:
                    total += self.store_chunks_for_meta(meta)
                except Exception:
                    self.logger.exception(
                        "Failed to store chunks for %s", meta.path
                    )
        return total
            
                
    async def insert_docs_from_meta(self, skipocr: bool = True):
        """Re-extract text for filemeta rows in the pool that have empty ``inhalt``.

        For ``doc`` filemeta, ensures a :class:`DBDoc` row exists for embedding FKs.

        Args:
        skipocr (bool): Wenn True, wird die OCR-Verarbeitung übersprungen, um Ressourcen zu sparen.
        """
        with self.Session() as session:
            stmt = sa.select(DBMeta).where(
                DBMeta.pool_id == self.pool.id,
                DBMeta.inhalt == "",
            )
            metalist = session.execute(stmt).scalars().all()
            afop = AsyncFileOperations(pool=self.pool, config=self.config, skip_ocr=skipocr)
            for dbmeta in metalist:
                _, content = await afop.tika_parse(dbmeta.path)
                dbmeta.inhalt = clean_document_text(content)
                
            session.flush()
            session.commit()
        
        

    def insert_pics_from_meta(self):
        """Liest die Metadaten aller Bilder aus dem angegebenen Pool aus,
        erstellt zugehörige DBPic-Objekte und fügt sie der DB hinzu.

        
        """
        with self.Session() as session:
            subq = sa.select(DBPic.id).where(DBPic.meta_id == DBMeta.id).exists()
            stmt = sa.select(DBMeta).where(
                DBMeta.pool_id == self.pool.id,
                DBMeta.ftype == "pic",
                ~subq
            )
            metalist = session.execute(stmt).scalars().all()
            afop = AsyncFileOperations(pool=self.pool, config=self.config, skip_ocr=True)
            for dbmeta in metalist:
                dbmeta.pic = afop.DBPic_from_DBMeta(dbmeta)
            session.flush()
            session.commit()
                

    def update_thumbs_no_heic(self)->List[int]:
        """Update missing thumbnails for pictures in the pool that are not HEICs."""
        with self.Session() as session:
            stmt = (
                sa.select(DBMeta.id, DBMeta.path)
                .join(DBMeta.pic)
                .where(DBMeta.pool_id == self.pool.id)
                .where(DBMeta.thumbarray.is_(None))
                .where(DBMeta.suffix.not_in([".HEIC", ".heic"]))
            )
        
            piclist = [
                (os.path.join(self.docroot, row.path), row.id)
                for row in session.execute(stmt)
            ]
        resizer = _dali_resizer(pipe_batch_size=10, num_threads=4)
        grespics, greslabels, gerrfiles, gerrlabels = resizer.resize_pics(piclist, batch_size=100, use_PIL=False)
        thumblist = list(zip(grespics, greslabels))
        self.update_thumbs(thumblist)
        return gerrlabels
        
        

    def update_thumbs_no_thumb(self)-> List[int]:
        """Update missing thumbnails for pictures in the pool."""
        with self.Session() as session:
            stmt = (
                sa.select(DBMeta.id, DBMeta.path)
                .join(DBMeta.pic)
                .where(DBMeta.pool_id == self.pool.id)
                .where(DBMeta.thumbarray.is_(None))
            )
            piclist = [
                (os.path.join(self.docroot, row.path), row.id)
                for row in session.execute(stmt)
            ]
        resizer = _dali_resizer(pipe_batch_size=1, num_threads=4,)
        grespics, greslabels, gerrfiles, gerrlabels = resizer.resize_pics(piclist, batch_size=1, use_PIL=True)
        thumblist = list(zip(grespics, greslabels))
        self.update_thumbs(thumblist)
        return gerrlabels


    def update_thumbs(self, thumblist: List[Tuple[np.ndarray, int]]):
        """Update thumbnails on filemeta rows. *id* is ``filemeta.id``."""
        with self.Session() as session:
            for thumb, meta_id in thumblist:
                meta = session.get(DBMeta, meta_id)
                if meta is None:
                    self.logger.warning("No DBMeta row for id %s", meta_id)
                    continue
                meta.thumbarray = thumb
                meta.set_phash()
            session.flush()
            session.commit()

    def update_phashes(self) -> int:
        """Compute missing perceptual hashes for rows that already have thumbnails."""
        with self.Session() as session:
            metas = session.scalars(
                sa.select(DBMeta)
                .where(DBMeta.pool_id == self.pool.id)
                .where(DBMeta.thumbarray.is_not(None))
                .where(DBMeta.phash.is_(None))
            ).all()
            for meta in metas:
                meta.set_phash()
            count = len(metas)
            session.flush()
            session.commit()
        return count

    def update_thumb_for_pic(self, pic_id: int, rel_path: str) -> bool:
        """Generate a thumbnail for one picture row using PIL."""
        from .thumbnail import make_thumbnail_array

        with self.Session() as session:
            pic = session.get(DBPic, pic_id)
            if pic is None:
                self.logger.warning("No DBPic row for id %s", pic_id)
                return False
            meta_id = pic.meta_id

        full_path = os.path.join(self.docroot, rel_path)
        try:
            thumb = make_thumbnail_array(full_path)
        except Exception as exc:
            self.logger.warning(
                "Thumbnail generation failed for %s (pic id %s): %s",
                rel_path,
                pic_id,
                exc,
            )
            return False
        self.update_thumbs([(thumb, meta_id)])
        return True

    def update_thumb_for_doc(self, meta_id: int, rel_path: str) -> bool:
        """Generate a thumbnail for one document row."""
        from .document_thumbnail import make_document_thumbnail

        full_path = os.path.join(self.docroot, rel_path)
        try:
            thumb = make_document_thumbnail(full_path)
        except Exception as exc:
            self.logger.warning(
                "Document thumbnail generation failed for %s (meta id %s): %s",
                rel_path,
                meta_id,
                exc,
            )
            return False
        self.update_thumbs([(thumb, meta_id)])
        return True

    def update_thumbs_doc(self) -> List[int]:
        """Update thumbnails for documents in the pool that have no thumbnail."""
        with self.Session() as session:
            rows = list(
                session.execute(
                    sa.select(DBMeta.id, DBMeta.path)
                    .where(DBMeta.pool_id == self.pool.id)
                    .where(DBMeta.ftype == FTYPE.DOC)
                    .where(DBMeta.thumbarray.is_(None))
                )
            )

        failed: list[int] = []
        for row in rows:
            if not self.update_thumb_for_doc(row.id, row.path):
                failed.append(row.id)
        return failed

    def update_thumb_for_path(self, rel_path: str) -> bool:
        """Generate a missing thumbnail for one indexed picture path."""
        with self.Session() as session:
            stmt = (
                sa.select(DBPic.id)
                .join(DBPic.meta)
                .where(
                    DBMeta.pool_id == self.pool.id,
                    DBMeta.path == rel_path,
                    DBMeta.thumbarray.is_(None),
                )
            )
            pic_id = session.scalars(stmt).first()
        if pic_id is None:
            return False
        return self.update_thumb_for_pic(pic_id, rel_path)

class AsyncFileOperations(AsyncTikaClient):
    """Async file parsing helpers built on top of ``tika_client``."""
    def __init__(
        self,
        pool: DBPool,
        config: Config | None = None,
        skip_ocr: bool = True,
        timeout: float = 30,
        compress=True,
    ):
        config = resolve_config(config)
        self.config = config
        self.logger = config.logger
        self.docroot = config.resolved_docroot()
        self.tika_url = config.tika_noocr_url if skip_ocr else config.tika_ocrurl
        self.pool = pool
        super().__init__(self.tika_url, timeout=timeout, compress=compress)
        
    async def __aenter__(self) -> AsyncFileOperations:
        """Enter the TikaClient context."""
        return self
    
    def dir_tree(self, rel_path: str) -> Generator[str]:
        """Durchläuft rekursiv alle Dateien im Verzeichnisbaum, 
        extrahiert die Metadaten und Inhalte und erstellt DB-Objekte."""
        abspath = os.path.join(self.docroot,rel_path)
        assert os.path.exists(abspath), f"Directory {abspath} does not exist"
        for dirpath, _, filenames in os.walk(abspath):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                yield os.path.relpath(filepath, self.docroot)
    
    async def tika_parse(self, rel_path: str) -> Tuple[DBMeta|None, str]:
        """Parse one file with Tika and convert metadata into ``DBMeta`` fields."""
        filepath = os.path.join(self.docroot, rel_path)
        try:
            parsed: list[TikaResponse] = await self.rmeta.as_text.from_file(Path(
                filepath))
            meta: dict = parsed[0].data
            meta = json.loads(json.dumps(parsed[0].data).replace("\\u0000", "null"))
            content = meta.pop(TikaKey.Content, "")  # type: ignore
            if isinstance(content, list):
                content = content[0] if content else ""
            if not isinstance(content, str):
                content = str(content) if content else ""
            content = content.strip("\n\r")
            content = clean_document_text(content)
        except Exception as e:
            self.logger.error("Error parsing file %s: %r", filepath, e)
            meta = {}
            content = ""
        
        meta["file"] = filepath
        # meta["fdate"] = datetime.fromtimestamp(
        #     os.path.getmtime(filepath)
        # )
        fsize = os.path.getsize(filepath)
        
        path = rel_path
        fdate: datetime = datetime.fromtimestamp(os.path.getmtime(filepath))
        doc_cdate_str: str|None|list[str] = meta.get("dcterms:created", None)
        if isinstance(doc_cdate_str, list):
            doc_cdate_str = doc_cdate_str[0]
        if doc_cdate_str is None:
            doc_cdate = fdate
        else:  
            try:
                doc_cdate: datetime = datetime.fromisoformat(doc_cdate_str) # type: ignore
            except Exception as e:
                self.logger.error("Error %s parsing document creation date %s", e, doc_cdate_str)
                doc_cdate = fdate
        content_length = meta.get("Content-Length", 0)
        if isinstance(content_length, list):
            content_length: int = content_length[0]
        
        suffix = os.path.splitext(path)[1]
        ftype = self.config.get_ftype(suffix)
        with open(filepath, "rb") as f:
            sha256 = file_digest(f, "sha256")
        try:
            dbmeta = DBMeta(
                pool_id=self.pool.id,
                path=path,
                fname=os.path.basename(path),
                fdate=fdate,
                sort_date=doc_cdate,
                fsize=fsize,
                clength=content_length,
                ctype=meta.get("Content-Type", "unk/unk"),
                md_keys=list(meta.keys()),
                suffix=suffix,
                ftype=ftype,
                sha256=sha256.digest(),
                meta_data=meta,
                inhalt=content,
            )
        except Exception as e:
            self.logger.error("Error %s creating DBMeta fpath %s", e, rel_path)
            return None, content 
        else:
            return dbmeta, content 
    
    
    
    def DBPic_from_DBMeta(self, dbmeta: DBMeta) -> DBPic:
        """Build a ``DBPic`` row from image metadata and XMP tags."""
        assert dbmeta.ftype == "pic", "DBMeta object must have ftype 'pic'"
        
        xmp = xmpu.file_to_dict(os.path.join(self.docroot,dbmeta.path))
        xmpdict = dict(
            [
                (a, b.replace("\\u0000", "null"))
                for data in xmp.values()
                for a, b, _ in data
            ]
        )

        dbpic = DBPic(xmp=xmpdict)
        dbpic.meta_id = dbmeta.id
        
        
        return dbpic
           
        
    async def file_to_db(self, rel_path: str) ->  DBMeta|None:
        """Create ``DBMeta`` and optional typed child rows for a file path."""
        dbmeta, content = await self.tika_parse(rel_path)
        if dbmeta is None:
            return None

        #if dbmeta.ftype == "doc":
        #    dbmeta.doc = DBDoc()
        if dbmeta.ftype == "pic":
            dbmeta.pic = self.DBPic_from_DBMeta(dbmeta)
        if dbmeta.ftype == "vid":
            dbmeta.vid = DBVid()
        return dbmeta
