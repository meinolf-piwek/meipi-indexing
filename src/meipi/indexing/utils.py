"""General utilities for indexing."""

import io
from itertools import batched
from pathlib import Path
from tqdm.auto import tqdm
import sqlalchemy as sa
from PIL import Image
from meipi.indexing.preprocess import DALIImageResizer, dbpic_from_dbmeta
from meipi.indexing.db import DBMeta, DBPic, pgEngine
from meipi.indexing import appconf

image_pattern = appconf.picsuf
image_root = Path(appconf.docroot + "Bilder/")
engine = pgEngine(appconf.db_conn_string)
# Base.metadata.create_all(engine.engine)


def insert_pics_from_meta(pool):
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


def read_no_heic():
    with engine.Session() as session:
        stmt = (
            sa.select(DBPic.id, DBPic.path)
            .where(DBPic.thumbarray == None)
            .where(DBPic.suffix.not_in([".HEIC", ".heic"]))
        )
    return [(appconf.docroot + x.path, x.id) for x in session.execute(stmt)]


def read_pic_no_thumb():
    with engine.Session() as session:
        stmt = sa.select(DBPic.id, DBPic.path).where(DBPic.thumbarray == None)
    return [(appconf.docroot + x.path, x.id) for x in session.execute(stmt)]


def resize_pics(piclist, batch_size, pipe_batch_size, use_PIL):
    image_resizer = DALIImageResizer(
        pipe_batch_size=pipe_batch_size, num_threads=4, use_PIL=use_PIL
    )
    batches = batched(piclist, batch_size)
    grespics, greslabels, gerrfiles, gerrlabels = ([], [], [], [])
    for batch in tqdm(batches, total=(len(piclist) // batch_size)):
        files, labels = zip(*batch)
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
