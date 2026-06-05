"""Konfiguration der App

Neben der hier definierten default-Konfiguration,
können die Werte auch über eine .env-Datei oder direkt über Umgebungsvariablen überschrieben werden.
Die .env-Datei sollte im Root-Verzeichnis der App liegen und den Namen "config.env" tragen.

Die globale Instanz ``appconf`` wird einmal beim Import von ``meipi.indexing`` erzeugt. Felder können
zur Laufzeit gesetzt werden (z. B. ``--docroot`` in der CLI) oder per ``reload_appconf()`` aus
``config.env`` / ``MEIPI_CONFIG_ENV`` neu geladen werden. Bestehende ``DBOperations``-Instanzen
behalten danach ihre alte Engine (``search_path``) — neu anlegen nach einem Reload.

Beispiel für eine ``.env-Datei``::

    IND_PG_HOST=localhost                   #PostgreSQL Host
    IND_PG_PORT=5432                        #PostgreSQL Port
    IND_PG_USER=postgres                    #PostgreSQL Username
    IND_PG_DATABASE=postgres                #PostgreSQL Database Name
    IND_PG_API_KEY=pg-docker                #API-Key-Name für das DB-Passwort im Keyring
    IND_DOCROOT=/home/rslsync/folders/  #Dokumenten-Root-Verzeichnis
    IND_LOGGER_NAME=sqlalchemy.engine          #Name des Loggers für SQLAlchemy
    IND_DOCSUF='{
        ".pdf",
        ".txt",
        ".md",
        ".docx",
        ".doc",
        ".html",
        ".htm",
        ".epub",
        ".odt",
    }  #zulässige Dokumentenerweiterungen
    IND_PICSUF='{
        ".jpg",
        ".jpeg",
        ".bmp",
        ".png",
        ".heic",
        ".tiff",
        ".tif"
    }  #zulässige Bild-Dateiendungen
    IND_VIDSUF='{
        ".mov",
        ".vob",
        ".mkv",
        ".avi",
        ".mp4",
        ".mcf"
    }  #zulässige Video-Dateiendungen
    IND_LOGLEVEL=20  #Log-Level (z.B. 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL)

Das DB-Passwort wird aus dem Keyring geholt, der API-Key-Name
kann über die Umgebungsvariable IND_PG_API_KEY konfiguriert werden (Standard: "pg-docker").

Alternatives Env-File vor dem Start setzen::

    export MEIPI_CONFIG_ENV=/path/to/config.env
    meipi-index schema-info
"""

from typing import Set
import logging
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import MetaData, URL as saURL
#from pillow_heif import register_heif_opener
#from PIL import ImageFile

class FTYPE():
    DOC:str = "doc"
    PIC:str = "pic"
    VID:str = "vid"
    UNK:str = "unk"


class Config(BaseSettings):
    """Enthält die Konfiguration der App

    Zur Laufzeit gilt ``appconf``; Felder können in-process geändert werden (z. B. CLI-Overrides).
    Das PostgreSQL-Schema wird über ``pg_schema`` (``search_path``) gesteuert, nicht über qualifizierte
    Tabellennamen in den ORM-``Table``-Objekten.
    """
    envfile: str = Field(default="config.env", description="Path used when this config was loaded")
    model_config = SettingsConfigDict(
        env_file="config.env",
        env_file_encoding="utf-8",
        env_prefix="IND_",
        case_sensitive=False,
    )
    pg_host: str = Field(default="localhost")
    pg_port: str = Field(default="5432")
    pg_user: str = Field(default="postgres")
    pg_passwd: str = Field(default="postgres")
    pg_database: str = Field(default="postgres")
    pg_schema: str = Field(default="public")
    pg_api_key: str = Field(default="pg-docker")
    tika_noocr_url: str = Field(default="http://localhost:9998")
    tika_ocrurl: str = Field(default="http://localhost:9997")
    docroot: str = Field(
        default=".",
        description="Filesystem root; file paths in the DB are relative to this directory",
    )
    docsuf: Set[str] = Field(default={".pdf",".txt",".md",".docx",".doc",
        ".html",".htm",".epub",".odt",})
    picsuf: Set[str] = Field(default={".jpg", ".jpeg", ".bmp", ".png", ".heic", ".tiff", ".tif"})
    vidsuf: Set[str] = Field(default={".mov", ".vob", ".mkv", ".avi", ".mp4", ".mcf"})
    logger_name: str = "sqlalchemy.engine"
    loglevel: int = logging.INFO
        
    @property
    def logger(self):
        return logging.getLogger(self.logger_name)

    @property
    def metadata(self) -> MetaData:
        """Shared ORM ``MetaData`` (tables are unqualified; schema via ``pg_schema`` / search_path)."""
        from meipi.indexing.model import orm_metadata

        return orm_metadata

    def resolved_docroot(self) -> str:
        """Absolute path to the configured filesystem root."""
        return os.path.abspath(self.docroot)

    @classmethod
    def load(cls, envfile: str = "config.env") -> "Config":
        """Load settings from *envfile* (env vars still override file values)."""
        return cls(_env_file=envfile, envfile=envfile)

    def db_passwd_from_keyring(self) -> str:
        """DB password from SecretService keyring when D-Bus is available, else ``pg_passwd``."""
        if not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
            return self.pg_passwd
        try:
            import keyring
            from keyring.backends.SecretService import Keyring as SecretServiceKeyring

            keyring.set_keyring(SecretServiceKeyring())
            pwd = keyring.get_password("API-Keys", self.pg_api_key)
            if pwd is not None:
                return pwd
            self.logger.warning(
                "Kein Passwort für API-Key '%s' im Keyring gefunden, verwende pg_passwd aus Konfiguration",
                self.pg_api_key,
            )
        except Exception as exc:
            self.logger.warning(
                "Keyring nicht verfügbar (%s), verwende pg_passwd aus Konfiguration",
                exc,
            )
        return self.pg_passwd
            

    @property
    def db_conn_URL(self) -> saURL:
        """DB-Verbindungsstring aus den Konfigurationswerten zusammensetzen"""
        pg_passwd = self.db_passwd_from_keyring()
        return  saURL.create(
        "postgresql+psycopg",
        username=self.pg_user,
        password=pg_passwd,
        host=self.pg_host,
        port=int(self.pg_port),
        database=self.pg_database,
)
            

    def get_ftype(self, suf: str) -> str:
        """Gibt den konfigurierten Dateityp zurück, basierend auf der Dateiendung"""
        _suf = suf.lower()
        if _suf in self.docsuf:
            return FTYPE.DOC
        elif _suf in self.picsuf:
            return FTYPE.PIC
        elif _suf in self.vidsuf:
            return FTYPE.VID
        else:
            return FTYPE.UNK
            

def resolve_config(config: Config | None) -> Config:
    """Return *config* or the process-wide ``appconf``."""
    if config is None:
        import meipi.indexing as indexing_pkg

        return indexing_pkg.appconf
    return config


def reload_appconf(envfile: str | None = None) -> Config:
    """Reload the process-wide ``appconf`` from disk into the existing instance.

    Uses *envfile*, else ``MEIPI_CONFIG_ENV``, else ``appconf.envfile``.
    In-place update keeps ``from meipi.indexing import appconf`` bindings valid.
    Create new ``DBOperations`` afterwards if connection settings changed.
    """
    import meipi.indexing as indexing_pkg

    path = envfile or os.environ.get("MEIPI_CONFIG_ENV") or indexing_pkg.appconf.envfile
    fresh = Config.load(path)
    current = indexing_pkg.appconf
    for field_name in Config.model_fields:
        setattr(current, field_name, getattr(fresh, field_name))
    return current


def install_appconf(config: Config) -> None:
    """Replace the process-wide ``appconf``."""
    import sys

    import meipi.indexing as indexing_pkg

    indexing_pkg.appconf = config  # type: ignore[misc, assignment]

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        name = getattr(mod, "__name__", "")
        if name.startswith("meipi.indexing") and hasattr(mod, "appconf"):
            mod.appconf = config #type: ignore[assignment]
