"""Package für Indizierung von Dokumenten, Bildern und Videos

Dokumente und Bilder können vom File-System gelesen werden und 
ihre Metadaten sowie Vector-Einbettungen werden in einer Postgres-DB gespeichert
"""

from .config import Config


__version__ = "0.0.1"
appconf:Config = Config.from_env_file()
