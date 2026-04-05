"""Konfiguration der App

Neben der hier definierten default-Konfiguration,
können die Werte auch über eine .env-Datei oder direkt über Umgebungsvariablen überschrieben werden.
Die .env-Datei sollte im Root-Verzeichnis der App liegen und den Namen "config.env" tragen.

Beispiel für eine ``.env-Datei``::

    PG_HOST=localhost                   #PostgreSQL Host
    PG_PORT=5432                        #PostgreSQL Port
    PG_USER=postgres                    #PostgreSQL Username
    PG_DATABASE=postgres                #PostgreSQL Database Name
    PG_API_KEY=pg-docker                #API-Key-Name für das DB-Passwort im Keyring
    IND_DATADIR=./data                  #Datenverzeichnis
    IND_DOCROOT=/home/rslsync/folders/  #Dokumenten-Root-Verzeichnis
    IND_DOCSUF=.pdf,.txt,.md,.docx,.doc,.html,.htm,.epub,.odt  #zulässige Dokumentenerweiterungen
    IND_PICSUF=.jpg,.jpeg,.bmp,.png,.heic,.tiff,.tif  #zulässige Bild-Dateiendungen
    IND_VIDSUF=.mov,.vob,.mkv,.avi,.mp4,.mcf  #zulässige Video-Dateiendungen

Das DB-Passwort wird aus dem Keyring geholt, der API-Key-Name
kann über die Umgebungsvariable PG_API_KEY konfiguriert werden (Standard: "pg-docker").
"""

import os
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Sequence
from dotenv import load_dotenv
import keyring
from keyring.backends.SecretService import Keyring as SecretServiceKeyring


@dataclass
class Config:
    """Enthält die Konfiguration der App

    Die Konfiguration kann entweder über Umgebungsvariablen oder über eine .env-Datei bereitgestellt werden.
    Die .env-Datei sollte im Root-Verzeichnis der App liegen und den Namen "config.env" tragen.
    """

    db_conn_string: str
    datadir: str
    docroot: str
    logger: logging.Logger = logging.Logger("sqlalchemy.engine", logging.INFO)
    docsuf: Sequence[str] = (
        ".pdf",
        ".txt",
        ".md",
        ".docx",
        ".doc",
        ".html",
        ".htm",
        ".epub",
        ".odt",
    )
    picsuf: Sequence[str] = (".jpg", ".jpeg", ".bmp", ".png", ".heic", ".tiff", ".tif")
    vidsuf: Sequence[str] = (".mov", ".vob", ".mkv", ".avi", ".mp4", ".mcf")

    @staticmethod
    def get_db_passwd() -> str|None:
        """DB Password aus Keyring holen, oder Default-Wert verwenden"""
        api_key = os.getenv("PG_API_KEY", "pg-docker")
        keyring.set_keyring(SecretServiceKeyring())
        return keyring.get_password("API-Keys", api_key)

    @classmethod
    def from_env_file(
        cls, envfile: str = "config.env", override: bool = False, **kwargs
    ):
        """Lesen der Konfiguration aus Umgebung oder env-file"""
        with suppress(FileNotFoundError):
            load_dotenv(envfile, override=override)
        host = os.getenv("PG_HOST", "localhost")
        port = os.getenv("PG_PORT", "5432")
        user = os.getenv("PG_USER", "postgres")
        database = os.getenv("PG_DATABASE", "postgres")
        password = cls.get_db_passwd()
        db_conn_string = (
            f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
        )
        datadir = os.getenv("IND_DATADIR", ".")
        docroot = os.getenv("IND_DOCROOT", "/home/rslsync/folders/")
        # TODO: Umwandlung testen
        docsufstr = os.getenv("IND_DOCSUF")
        docsuf = (
            tuple(s.strip() for s in docsufstr.split(",")) if docsufstr else cls.docsuf
        )
        picsufstr = os.getenv("IND_PICSUF")
        picsuf = (
            tuple(s.strip() for s in picsufstr.split(",")) if picsufstr else cls.picsuf
        )
        vidsufstr = os.getenv("IND_VIDSUF")
        vidsuf = (
            tuple(s.strip() for s in vidsufstr.split(",")) if vidsufstr else cls.vidsuf
        )
        return cls(
            db_conn_string=db_conn_string,
            datadir=datadir,
            docroot=docroot,
            docsuf=docsuf,
            picsuf=picsuf,
            vidsuf=vidsuf,
        )

    def get_ftype(self, suf: str) -> str|None:
        """Gibt den konfigurierten Dateityp zurück, basierend auf der Dateiendung"""
        _suf = suf.lower()
        if _suf in self.docsuf:
            return "doc"
        elif _suf in self.picsuf:
            return "pic"
        elif _suf in self.vidsuf:
            return "vid"
        else:
            return None
