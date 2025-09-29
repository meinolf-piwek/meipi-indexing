"""Konfiguration der App"""

import os
from typing import overload
import logging
from contextlib import suppress
from dataclasses import dataclass
from dotenv import load_dotenv
import keyring
from keyring.backends.SecretService import Keyring as SecretServiceKeyring


@dataclass
class Config:
    """Enthält die Konfiguration der App"""

    db_conn_string: str
    datadir: str
    docroot: str
    logger: logging.Logger = logging.Logger("sqlalchemy.engine", logging.INFO)
    docsuf: tuple[str] = (
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
    picsuf: tuple[str] = (
        ".jpg",
        ".jpeg",
        ".bmp",
        ".png",
        ".heic",
        ".tiff",
        ".tif"
    )
    vidsuf: tuple[str] = (
        ".mov",
        ".vob",
        ".mkv",
        ".avi",
        ".mp4",
        ".mcf"
    )

    @staticmethod
    def get_db_passwd() -> str:
        """Je nach Target anders machen!"""
        api_key = os.getenv("PG_API_KEY","pg-docker")
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
        docsuf = os.getenv("IND_DOCSUF",cls.docsuf)
        picsuf = os.getenv("IND_PICSUF", cls.picsuf)
        vidsuf = os.getenv("IND_VIDSUF", cls.vidsuf)
        return cls(db_conn_string=db_conn_string, 
                   datadir=datadir,
                   docroot=docroot,
                   docsuf = docsuf,
                   picsuf = picsuf,
                   vidsuf = vidsuf
                   )
    
    def get_ftype(self,suf:str)-> str:
        _suf = suf.lower()
        if _suf in self.docsuf:
            return "doc"
        elif _suf in self.picsuf:
            return "pic"
        elif _suf in self.vidsuf:
            return "vid"
        else:
            return None
        
            
