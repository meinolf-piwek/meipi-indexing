"""Hauptfunktion, nur temporär interessant.

Hier sollen später die Funktionen zum Einlesen von Dokumenten, 
Bildern und Videos in die DB aufgerufen werden.
"""

# TODO: Das hier soll später in eine CLI oder API umgewandelt werden.
from pathlib import Path
from itertools import batched
from pickle import dump
from tqdm import tqdm
from .preprocess import tika_parse
from . import appconf

# from .preprocess import get_DBMeta_from_file
# from .db import pgEngine, DBDoc, DBPic


def main():
    """Liest Dateien aus dem File-System, extrahiert Metadaten und Textinhalte
    und speichert sie in einer Pickle-Datei.
    """
    filelist = (
        str(root / file)
        for root, _, files in Path(appconf.docroot + "/mobile").walk()
        for file in files)
    
    batches = batched(filelist, n=1000)
    docs = []
    for batch in tqdm(batches):
        docs.extend(tika_parse(batch))
    with open("pickle.pkl", "wb") as f:
        dump(docs, f)
