"""General utilities for indexing.

    This module contains functions for reading metadata from the database, resizing images, and updating the database with thumbnails.
    """
import io
from itertools import batched
from pathlib import Path
from typing import List, Tuple
from tqdm.auto import tqdm
import sqlalchemy as sa
from PIL import Image
from .preprocess import DALIImageResizer, dbpic_from_dbmeta
from .db import DBMeta, DBPic, pgEngine
from . import appconf

type PicList = List[Tuple[str, int]]  # List of tuples (file path, id)
image_pattern = appconf.picsuf
image_root = Path(appconf.docroot + "Bilder/")
engine = pgEngine(appconf.db_conn_string)
# Base.metadata.create_all(engine.engine)


def insert_pics_from_meta(pool:str):
    """Liest die Metadaten aller Bilder aus dem angegebenen Pool aus,
    erstellt zugehörige DBPic-Objekte und fügt sie der DB hinzu.

    Args:
        pool (str): Frei wählbarer Name für den Datenpool, z.B. "Bilder", "Texte", etc.
    """
    with engine.Session() as session:
        stmt = sa.select(DBMeta).where(DBMeta.pool == pool)
        metalist = session.execute(stmt).scalars().all()
    piclist = [
        dbpic_from_dbmeta(x, docroot=appconf.docroot)
        for x in metalist
        if appconf.get_ftype(x.suffix) == "pic"
    ]
    with engine.Session() as session:
        session.add_all(piclist)
        session.flush()
        session.commit()


def read_no_heic()->PicList:
    """Liest die Pfade und ids aller Bilder aus der Datenbank, die noch keinen Thumbnail haben, aber keine HEICs sind."""
    with engine.Session() as session:
        stmt = (
            sa.select(DBPic.id, DBPic.path)
            .where(DBPic.thumbarray == None)
            .where(DBPic.suffix.not_in([".HEIC", ".heic"]))
        )
    return [(appconf.docroot + x.path, x.id) for x in session.execute(stmt)]


def read_pic_no_thumb()-> PicList:
    """Liest die Pfade und ids aller Bilder aus der Datenbank, die noch keinen Thumbnail haben"""
    with engine.Session() as session:
        stmt = sa.select(DBPic.id, DBPic.path).where(DBPic.thumbarray == None)
    return [(appconf.docroot + x.path, x.id) for x in session.execute(stmt)]


def resize_pics(piclist: PicList, batch_size: int, pipe_batch_size: int, use_PIL: bool)-> Tuple[
    List[bytes], List[int], List[str], List[int]]:
    """Erstellt Thumbnails für die Bilder in piclist

    Args:
        piclist (PicList): Liste von Paaren aus Dateipfad und id
        batch_size (int): Anzahl der Bilder, die in einem Batch verarbeitet werden sollen
        pipe_batch_size (int): Anzahl Bilder pro Batch, die an die DALI-Pipeline übergeben werden sollen
        use_PIL (bool): Ob die Thumbnails mit PIL erstellt werden sollen (True) oder mit DALI (False)

    Returns:
        Tuple[List[bytes], List[int], List[str], List[int]]: Vier Listen: 1. Thumbnails als Byte-Arrays, 
        2. zugehörige ids, 3. Pfad der fehlgeschlagene Bilder, 4. ids der fehlgeschlagenen Bilder   
    """
    image_resizer = DALIImageResizer(
        pipe_batch_size=pipe_batch_size, num_threads=4, use_PIL=use_PIL
    )
    batches = batched(piclist, batch_size)
    grespics, greslabels, gerrfiles, gerrlabels = ([], [], [], [])
    for batch in tqdm(batches, total=(len(piclist) // batch_size)):
        files, labels= zip(*batch)
        respics, reslabels, errfiles, errlabels = image_resizer.process(
            files=files, labels=labels, batch_size=batch_size, show_progress=False
        )
        grespics.extend(respics)
        greslabels.extend(reslabels)
        gerrfiles.extend(errfiles)
        gerrlabels.extend(errlabels)
    return grespics, greslabels, gerrfiles, gerrlabels


def update_thumbs(labels, thumbs):
    updlist = []
    for l, t in zip(labels, thumbs):
        buf = io.BytesIO()
        im = Image.fromarray(t)
        im.save(fp=buf, format="png")
        buf.seek(0)
        updlist.append({"id": l, "thumbnail": buf.read()})
    with engine.Session() as session:
        stmt = sa.update(DBPic)
        session.execute(stmt, updlist)
        session.commit()


def update_thumb_array(labels, thumbs):
    updlist = [{"id": l, "thumbarray": t} for l, t in zip(labels, thumbs)]
    with engine.Session() as session:
        session.execute(sa.update(DBPic), updlist)
        session.commit()
