"""Hauptfunktion, nur temporär interessant.

Hier sollen später die Funktionen zum Einlesen von Dokumenten, 
Bildern und Videos in die DB aufgerufen werden.
"""

# TODO: Das hier soll später in eine CLI oder API umgewandelt werden.
from pathlib import Path
from itertools import batched
from pickle import dump
from tqdm import tqdm
import asyncio

from .. import appconf, DBOperations, DBPool, AsyncFileOperations



async def read_meta_bulk(pool:DBPool, relpath: str, pickle: bool=False):
    """Liest Dateien aus dem File-System, extrahiert Metadaten und Textinhalte
    und speichert sie in einer Pickle-Datei.
    """
    metalist = []
    async with AsyncFileOperations(pool, appconf) as afop:
        files = afop.dir_tree(relpath)
        batches = batched(files, 500)
        for batch in tqdm(batches):
            tasklist = set()
            for file in batch:
                tasklist.add(asyncio.create_task(afop.file_to_db(file)))
            async for task in asyncio.as_completed(tasklist):
                dbmeta,_,_,_ = await task
                metalist.append(dbmeta)
    if pickle:
        with open(appconf.datadir+"metalist.pkl", "wb") as f:
            dump(metalist, f)
    return metalist

