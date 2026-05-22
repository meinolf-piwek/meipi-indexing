"""Package für Indizierung von Dokumenten, Bildern und Videos

Dokumente und Bilder können vom File-System gelesen werden und 
ihre Metadaten sowie Vector-Einbettungen werden in einer Postgres-DB gespeichert
"""
__all__ = ["Config", 
           "appconf",
           "DBPool",
           "DBOperations",
           "DBMeta", "DBDoc", "DBPic", "Base", "DBDinoV2Vector", "AsyncFileOperations"]
__version__ = "0.0.1"

from .config import Config
appconf:Config = Config()
from .operations import DBOperations, AsyncFileOperations
from .model import DBMeta, DBDoc, DBPic, Base, DBDinoV2Vector, DBPool





