"""Definiert die IO-Operationen für die App."""
import io
import os
from typing import List, Tuple, Generator, Sequence
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
from libxmp import utils as xmpu
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, Session
from .model import Base as ModelBase, DBMeta, DBPic, DBDoc, DBDinoV2Vector, IdList,DBPool
from .config import Config
from . import appconf



class DBOperations():
    """Enthält Methoden für die DB-Operationen der App, z.B. Einfügen von Bildern, Lesen von Pfaden, etc."""
    def __init__(self, pool: DBPool, config: Config = appconf,
        metadata: sa.MetaData = ModelBase.metadata,
        enginekwargs: dict|None = None,
        sessionkwargs: dict|None = None):
        
        enginekwargs = {} if not enginekwargs else enginekwargs
        sessionkwargs = {} if not sessionkwargs else sessionkwargs
        self.engine = sa.create_engine(config.db_conn_URL, logging_name="DBOperations",pool_logging_name=config.logger_name, **enginekwargs)
        self.config = config
        self.logger = config.logger
        self.pool = pool
        self.docroot = pool.rootpath
        self.metadata = metadata
        try:
            self.Session = sessionmaker(
                bind=self.engine, expire_on_commit=False, **sessionkwargs
            )
        except Exception as e:
            self.logger.error("Error creating PostgreSQL engine: %s", e)
            raise
        else:
            self.logger.info("PostgreSQL engine created successfully.")

    def create_tables(self):
        """Erstellt die Tabellen in der Datenbank, falls sie noch nicht existieren."""
        self.metadata.create_all(self.engine)
        
    def recreate_tables(self):
        """Recreate tables in the database."""
        self.metadata.drop_all(self.engine)
        self.metadata.create_all(self.engine)
        
    def get_session(self, **kwargs)-> Session:
        return self.Session(**kwargs)

    def bulk_insert(self, TableClass: type[ModelBase], data: list[dict]):
        """Insert a list of data into the database."""
        with self.get_session(expire_on_commit=False) as session:
            try:
                stmt = sa.insert(TableClass).values(data)
                session.execute(stmt)
            except Exception as e:
                self.logger.error("Error %s inserting", e)
                session.rollback()
                return False
            else:
                session.commit()
                self.logger.warning(
                    "Inserted data into %s table", TableClass.__tablename__
                )
                return True
    
    def meta_not_in_db(self,files:Sequence[Path]):
        with self.Session() as session:
            for file in files:
                path = str(file).replace(self.docroot, "")
                
            
            
    
      
    async def insert_docs_from_meta(self, pool: str, skipocr: bool = True):
        """Liest die Metadaten aller Dokumente aus dem angegebenen Pool aus,
        erstellt zugehörige DBDoc-Objekte und fügt sie der DB hinzu.

        Args:
        pool (str): Frei wählbarer Name für den Datenpool, z.B. "Bilder", "Texte", etc.
        skipocr (bool): Wenn True, wird die OCR-Verarbeitung übersprungen, um Ressourcen zu sparen.
        """
        with self.Session() as session:
            stmt = sa.select(DBMeta).where(
                DBMeta.pool_id == self.pool.id and
                DBMeta.ftype == "doc" and
                DBMeta.id.not_in(
                    sa.select(DBDoc.id)
                )
            )
            metalist = session.execute(stmt).scalars().all()
        afop = AsyncFileOperations(pool=self.pool, config=self.config, skip_ocr=skipocr)
        doclist = [
            await afop.DBdoc_from_DBMeta(x)
            for x in metalist
            #if self.config.get_ftype(x.suffix) == "doc"
        ] 
        with self.Session() as session:
            session.add_all(doclist)
            session.flush()
            session.commit()    
                
    def insert_pics_from_meta(self):
        """Liest die Metadaten aller Bilder aus dem angegebenen Pool aus,
        erstellt zugehörige DBPic-Objekte und fügt sie der DB hinzu.

        
        """
        with self.Session() as session:
            stmt = sa.select(DBMeta).where(
                DBMeta.pool_id == self.pool.id and
                DBMeta.ftype == "pic" and
                DBMeta.id.not_in(
                    sa.select(DBPic.id)
                )
            )
            metalist = session.execute(stmt).scalars().all()
        afop = AsyncFileOperations(pool=self.pool, config=self.config, skip_ocr=True)
        piclist = [
            afop.DBPic_from_DBMeta(x)
            for x in metalist
            #if self.config.get_ftype(x.suffix) == "pic"
        ] 
        with self.Session() as session:
            session.add_all(piclist)
            session.flush()
            session.commit()

    def read_no_heic(self)->IdList:
        """Liest die Pfade und ids aller Bilder aus der Datenbank, 
        die noch keinen Thumbnail haben, aber keine HEICs sind."""
        with self.Session() as session:
            stmt = (
                sa.select(DBPic.id, DBPic.meta.path)
                .where(DBPic.thumbarray == None)
                .where(DBPic.meta.suffix.not_in([".HEIC", ".heic"]))
            )
        return [(self.docroot + x.path, x.id) for x in session.execute(stmt)]

    def read_pic_no_thumb(self)-> IdList:
        """Liest die Pfade und ids aller Bilder aus der Datenbank, 
        die noch keinen Thumbnail haben"""
        with self.Session() as session:
            stmt = sa.select(DBPic.id, DBPic.meta.path).where(DBPic.thumbarray == None)
        return [(self.docroot + x.path, x.id) for x in session.execute(stmt)]

    def update_thumbs(self, labels, thumbs):
        updlist = []
        for l, t in zip(labels, thumbs):
            buf = io.BytesIO()
            im = Image.fromarray(t)
            im.save(fp=buf, format="png")
            buf.seek(0)
            updlist.append({"id": l, "thumbnail": buf.read()})
        with self.Session() as session:
            stmt = sa.update(DBPic)
            session.execute(stmt, updlist)
            session.commit()

    def update_thumb_array(self, labels, thumbs):
        updlist = [{"id": l, "thumbarray": t} for l, t in zip(labels, thumbs)]
        with self.Session() as session:
            session.execute(sa.update(DBPic), updlist)
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
        for dirpath, _, filenames in os.walk(os.path.join(self.docroot,rel_path)):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                yield filepath
    
    async def tika_parse(self, filepath: str) -> Tuple[DBMeta|None, str]:
        """Get metadata of a file using Tika. file, fdate und fsize kommen vom os."""
        try:
            parsed: list[TikaResponse] = await self.rmeta.as_text.from_file(Path(filepath))
            meta: dict = parsed[0].data
            meta = json.loads(json.dumps(parsed[0].data).replace("\\u0000", "null"))
            content = meta.pop(TikaKey.Content, "")  # type: ignore
        except Exception as e:
            self.logger.error("Error %s parsing file %s", e, filepath)
            meta = {}
            content = ""
        
        meta["file"] = filepath
        meta["fdate"] = datetime.fromtimestamp(
            os.path.getmtime(filepath)
        ).isoformat()
        meta["fsize"] = os.path.getsize(filepath)
        assert filepath.startswith(self.docroot)
        path = os.path.relpath(str(filepath),self.docroot)
        os_fdate: str = meta.get("fdate", datetime.now().isoformat())
        doc_cdate: str|None = meta.get("dcterms:created", None) # type: ignore
        content_length = meta.get("Content-Length", 0)
        if isinstance(content_length, list):
            content_length: int = content_length[0]
        if isinstance(doc_cdate, list):
            doc_cdate: str = doc_cdate[0]
        suffix = os.path.splitext(path)[1]
        ftype = self.config.get_ftype(suffix)
        with open(filepath, "rb") as f:
            sha256 = file_digest(f, "sha256")
        try:
            dbmeta = DBMeta(
                id = None, # type: ignore
                pool_id= self.pool.id,
                path=path,
                fname=os.path.basename(path),
                fdate=os_fdate,
                sort_date=os_fdate if doc_cdate is None else doc_cdate,
                fsize=meta.get("fsize", 0),
                clength=content_length,
                ctype=meta.get("Content-Type", "unk/unk"),
                md_keys=list(meta.keys()),
                suffix=suffix,
                ftype=ftype,
                sha256=sha256.digest(),
                meta_data=meta,
            )
        except Exception as e:
            self.logger.error("Error %s creating DBMeta fpath %s", e, filepath)
            return None, content 
        else:
            return dbmeta, content 
    
    async def DBdoc_from_DBMeta(self,dbmeta: DBMeta) -> DBDoc|None:
        """
        Der Inhalt wird mit Tika extrahiert.
        
        """
        assert dbmeta.ftype == "doc", "DBMeta object must have ftype 'doc'"
        
        _, inhalt = await self.tika_parse(self.docroot + dbmeta.path)
            
        dbdoc = DBDoc(
            id= DBMeta.id, # type: ignore
            meta = dbmeta, # type: ignore
            inhalt=inhalt, # type: ignore
        )
        
        return dbdoc
    
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
        dbpic = DBPic(
            id = dbmeta.id, # type: ignore
            meta=dbmeta,
            xmp=xmpdict,
        )
        return dbpic
           
        
    async def file_to_db(self, filepath: str) ->  Tuple[DBMeta|None, DBDoc|None,
                                        DBPic|None, None]:
        """Get DBMeta and DBDoc objects from file path."""
        dbmeta, content = await self.tika_parse(filepath)
        if dbmeta is None:
            return None, None, None, None
        dbdoc = None
        dbpic = None
        dbvid = None
        if dbmeta.ftype == "doc":
            dbdoc = DBDoc(
                id= dbmeta.id, # type: ignore
                meta = dbmeta,
                inhalt=content, # type: ignore
                )
        if dbmeta.ftype == "pic":
            dbpic = self.DBPic_from_DBMeta(dbmeta)
        if dbmeta.ftype == "vid":
            dbvid = None
        return dbmeta, dbdoc, dbpic, dbvid
