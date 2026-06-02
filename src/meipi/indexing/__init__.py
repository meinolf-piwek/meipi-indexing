"""Package für Indizierung von Dokumenten, Bildern und Videos

Dokumente und Bilder können vom File-System gelesen werden und 
ihre Metadaten sowie Vector-Einbettungen werden in einer Postgres-DB gespeichert.

``appconf`` wird beim ersten Import dieses Pakets geladen und ist danach unveränderlich.
Konfiguration ändern: ``MEIPI_CONFIG_ENV`` setzen oder ``config.env`` anpassen, dann den
Prozess neu starten (CLI, Streamlit, Notebook-Kernel).
"""
__all__ = ["Config", 
           "appconf",
           "reload_appconf",
           "DBPool",
           "DBOperations",
           "DBMeta", "DBDoc", "DBPic", "Base", "DBDinoV2Vector", "DBCatalog",
           "AsyncFileOperations",
           "FTYPE"]
__version__ = "0.0.1"

import os
from typing import Final

from .config import Config, FTYPE, reload_appconf

_envfile = os.environ.get("MEIPI_CONFIG_ENV", "config.env")
appconf: Final[Config] = Config.load(_envfile)

from .operations import DBOperations, AsyncFileOperations
from .model import DBMeta, DBDoc, DBPic, Base, DBDinoV2Vector, DBPool, DBCatalog
