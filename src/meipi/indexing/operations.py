"""Definiert die IO-Operationen für die App."""
import io
import os
from typing import List, Optional, Tuple, Generator, Sequence, Any
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
from sqlalchemy.orm import sessionmaker, Session
from .model import Base as ModelBase, DBMeta, DBPic, DBDoc, DBDinoV2Vector, DBVid, IdList,DBPool
from .config import Config
from . import appconf

def _dali_resizer(*args, **kwargs):
    from .picture import DALIImageResizer

    return DALIImageResizer(*args, **kwargs)


class DBOperations():
    """Enthält Methoden für die DB-Operationen der App, z.B. Einfügen von Bildern, Lesen von Pfaden, etc."""
    def __init__(
        self,
        pool_id: int | None = None,
        pool: DBPool | None = None,
        *,
        allow_no_pool: bool = False,
        #config: Config = appconf,
        enginekwargs: dict | None = None,
        sessionkwargs: dict | None = None,
    ):
        self.config = appconf
        enginekwargs = {} if not enginekwargs else enginekwargs
        sessionkwargs = {} if not sessionkwargs else sessionkwargs
        self.metadata = ModelBase.metadata
        connect_args={"options": f"-c search_path={self.metadata.schema}"}
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
                id=0, pool="default", rootpath="", description="Default pool"
            )
        else:
            raise ValueError("Either pool_id or pool must be provided")
        self.docroot = self.pool.rootpath

    def schema_info(self) -> dict[str, Any]:
        """Get the schema information of the database."""
        tables = {}
        with self.Session() as session:
            cat = sa.Table('pg_tables', sa.MetaData(schema='pg_catalog'), autoload_with=self.engine)
            for tname in session.scalars(sa.select(cat.c.tablename).where(cat.c.schemaname == self.metadata.schema)):
                table = sa.Table(tname, self.metadata,autoload_with=self.engine)
                stmt = sa.select(sa.func.count()).select_from(table)
                tables[tname] = session.scalars(stmt).one()
        return {"schema":self.metadata.schema, "tables": tables} 
        
    def get_pool(self, pool_id: int)-> DBPool:
        """Get the pool with the given id."""
        with self.Session() as session:
            self.pool = session.get(DBPool, pool_id)
            if self.pool is None:
                raise ValueError(f"Pool with id {pool_id} not found")
            return self.pool

    def create_pool(self, pool: DBPool):
        with self.Session() as session:
            session.add(pool)  
            session.commit()
            session.refresh(pool)
            self.pool = pool
            self.docroot = pool.rootpath
            self.logger.info("Pool created with id %s", pool.id)
            return pool
    
    def create_tables(self, entities:Optional[Sequence[type[ModelBase]]]=None):
        """Erstellt die Tabellen in der Datenbank, falls sie noch nicht existieren."""
        if not entities:
            tables = None
        else:
            tables = [entity.__tablename__ for entity in entities]
        self.metadata.create_all(self.engine, tables=tables)
        
    def recreate_tables(self, entities:Optional[Sequence[type[ModelBase]]]=None):
        """Recreate tables in the database."""
        if not entities:
            tables = None
        else:
            tables = [entity.__tablename__ for entity in entities]
        self.metadata.drop_all(self.engine, tables=tables)
        self.metadata.create_all(self.engine, tables=tables)
        
         
    def clear_pool(self):
        """Löscht alle Daten aus dem angegebenen Pool."""
        with self.Session() as session:
            session.query(DBMeta).filter(DBMeta.pool_id == self.pool.id).delete()
            session.flush()
            session.commit()
            
                
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
                dbmeta.inhalt = content
                if dbmeta.ftype == "doc" and dbmeta.doc is None:
                    dbmeta.doc = DBDoc()
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
        """Update the thumbnails and perceptual hashes for the pictures in the pool that are not HEICs."""
        with self.Session() as session:
            stmt = (
                sa.select(DBPic.id, DBMeta.path)
                .join(DBPic.meta)
                .where(DBMeta.pool_id == self.pool.id)
                .where(DBPic.thumbarray.is_(None))
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
        """Update the thumbnails and perceptual hashes for the pictures in the pool that have no thumbnail."""
        with self.Session() as session:
            stmt = (
                sa.select(DBPic.id, DBMeta.path)
                .join(DBPic.meta)
                .where(DBMeta.pool_id == self.pool.id)
                .where(DBPic.thumbarray.is_(None))
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
        """Update the thumbnails and perceptual hashes for the given list of pictures."""
        with self.Session() as session:
            for thumb, id in thumblist:
                pic = session.get(DBPic, id)
                if pic is None:
                    self.logger.warning("No DBPic row for id %s", id)
                    continue
                pic.thumbarray = thumb
                pic.set_phash()
            session.flush()
            session.commit()

class AsyncFileOperations(AsyncTikaClient):
    """Enthält Methoden für die Datei-Operationen der App, z.B. Einlesen von Bildern, Berechnen von Hashes, etc."""
    def __init__(self, pool:DBPool, config: Config = appconf, 
            skip_ocr: bool = True, timeout: float = 30, compress=True):
        self.config = config
        self.logger = config.logger
        self.docroot = pool.rootpath
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
        """Get metadata of a file using Tika. file, fdate und fsize kommen vom os."""
        filepath = os.path.join(self.docroot, rel_path)
        try:
            parsed: list[TikaResponse] = await self.rmeta.as_text.from_file(Path(
                filepath))
            meta: dict = parsed[0].data
            meta = json.loads(json.dumps(parsed[0].data).replace("\\u0000", "null"))
            content = meta.pop(TikaKey.Content, "")  # type: ignore
        except Exception as e:
            self.logger.error("Error %s parsing file %s", e, filepath)
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
                #id = None, # type: ignore
                pool_id= self.pool.id,
                path=path,
                fname=os.path.basename(path),
                fdate=fdate,
                sort_date= doc_cdate,
                fsize=fsize,
                clength=content_length,
                ctype=meta.get("Content-Type", "unk/unk"),
                md_keys=list(meta.keys()),
                suffix=suffix,
                ftype=ftype,
                sha256=sha256.digest(),
                meta_data=meta,
            )
        except Exception as e:
            self.logger.error("Error %s creating DBMeta fpath %s", e, rel_path)
            return None, content 
        else:
            return dbmeta, content 
    
    
    
    def DBPic_from_DBMeta(self, dbmeta: DBMeta) -> DBPic:
        """Get DBPic object from DBMeta object."""
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
        """Get DBMeta (and optional child rows) from file path."""
        dbmeta, content = await self.tika_parse(rel_path)
        if dbmeta is None:
            return None

        dbmeta.inhalt = content
        if dbmeta.ftype == "doc":
            dbmeta.doc = DBDoc()
        if dbmeta.ftype == "pic":
            dbmeta.pic = self.DBPic_from_DBMeta(dbmeta)
        if dbmeta.ftype == "vid":
            dbmeta.vid = DBVid()
        return dbmeta
