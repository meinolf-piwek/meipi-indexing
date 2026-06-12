"""Package für Indizierung von Dokumenten, Bildern und Videos

Dokumente und Bilder können vom File-System gelesen werden und 
ihre Metadaten sowie Vector-Einbettungen werden in einer Postgres-DB gespeichert.

``appconf`` wird beim ersten Import geladen. Zur Laufzeit einzelne Felder setzen oder
``install_appconf()`` für einen komplett neuen ``Config``-Satz. Änderungen an der Env-Datei
erfordern einen Prozess-Neustart.
"""
__all__ = ["Config", 
           "appconf",
           "DBPool",
           "DBOperations",
           "DBMeta", "DBDoc", "DBPic", "Base", "DBDinoV2Vector", "DBBgeM3Vector", "DBCatalog",
           "AsyncFileOperations",
           "DocumentChunk",
           "FTYPE"]
__version__ = "0.0.1"

from .config import CONFIG_PATH, Config, FTYPE

appconf: Config = Config()

from .operations import DBOperations, AsyncFileOperations
from .model import DBMeta, DBDoc, DBPic, Base, DBDinoV2Vector, DBBgeM3Vector, DBPool, DBCatalog
from .document_chunks import DocumentChunk
