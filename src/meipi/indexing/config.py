"""Konfiguration der App

Neben der hier definierten default-Konfiguration,
können die Werte auch über eine .env-Datei oder direkt über Umgebungsvariablen überschrieben werden.
Die .env-Datei sollte im Root-Verzeichnis der App liegen und den Namen "config.env" tragen.

Die globale Instanz ``appconf`` wird einmal beim Import von ``meipi.indexing`` erzeugt und ist danach
unveränderlich. Änderungen an ``config.env``, Umgebungsvariablen oder ``MEIPI_CONFIG_ENV`` wirken
erst nach einem vollständigen Neustart des Python-Prozesses (CLI erneut ausführen, Notebook-Kernel
neu starten, Streamlit-App neu starten).

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
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL as saURL
import keyring
from keyring.backends.SecretService import Keyring as SecretServiceKeyring
#from pillow_heif import register_heif_opener
#from PIL import ImageFile

class FTYPE():
    DOC:str = "doc"
    PIC:str = "pic"
    VID:str = "vid"
    UNK:str = "unk"


class Config(BaseSettings):
    """Enthält die Konfiguration der App

    Instanzen sind unveränderlich (frozen). Erzeugen Sie für andere Env-Dateien eine neue
    ``Config`` nur in Tests oder vor dem ersten Import von ``meipi.indexing``; zur Laufzeit
    gilt ausschließlich ``appconf``.
    """
    envfile: str = Field(default="config.env", description="Path used when this config was loaded")
    model_config = SettingsConfigDict(
        env_file="config.env",
        env_file_encoding="utf-8",
        env_prefix="IND_",
        case_sensitive=False,
        frozen=True,
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
    docsuf: Set[str] = Field(default={".pdf",".txt",".md",".docx",".doc",
        ".html",".htm",".epub",".odt",})
    picsuf: Set[str] = Field(default={".jpg", ".jpeg", ".bmp", ".png", ".heic", ".tiff", ".tif"})
    vidsuf: Set[str] = Field(default={".mov", ".vob", ".mkv", ".avi", ".mp4", ".mcf"})
    logger_name: str = "sqlalchemy.engine"
    loglevel: int = logging.INFO
        
    @property
    def logger(self):
        return logging.getLogger(self.logger_name)

    @classmethod
    def load(cls, envfile: str = "config.env") -> "Config":
        """Load settings from *envfile* (used at package import, not for hot reload)."""
        return cls(envfile=envfile)

    def db_passwd_from_keyring(self) -> str:
        """DB Password aus Keyring holen, oder Default-Wert verwenden."""
        keyring.set_keyring(SecretServiceKeyring())
        pwd = keyring.get_password("API-Keys", self.pg_api_key)
        if pwd is None:
            self.logger.warning(
                "Kein Passwort für API-Key '%s' im Keyring gefunden, verwende Default-Wert",
                self.pg_api_key,
            )
            return self.pg_passwd
        return pwd
            

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
            

def reload_appconf() -> None:
    """Configuration cannot be reloaded in-process."""
    raise RuntimeError(
        "appconf cannot be reloaded. Set MEIPI_CONFIG_ENV or edit config.env, "
        "then restart the Python process (re-run CLI, restart Streamlit, or reload the notebook kernel)."
    )
