"""Konfiguration der App

Neben der hier definierten default-Konfiguration,
können die Werte auch über eine .env-Datei oder direkt über Umgebungsvariablen überschrieben werden.
Die .env-Datei sollte im Root-Verzeichnis der App liegen und den Namen "config.env" tragen.

Beispiel für eine ``.env-Datei``::

    IND_PG_HOST=localhost                   #PostgreSQL Host
    IND_PG_PORT=5432                        #PostgreSQL Port
    IND_PG_USER=postgres                    #PostgreSQL Username
    IND_PG_DATABASE=postgres                #PostgreSQL Database Name
    IND_PG_API_KEY=pg-docker                #API-Key-Name für das DB-Passwort im Keyring
    IND_DATADIR=./data                  #Datenverzeichnis
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
"""

from typing import Set
import logging
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL as saURL
import keyring
from keyring.backends.SecretService import Keyring as SecretServiceKeyring
from pillow_heif import register_heif_opener
from PIL import ImageFile

register_heif_opener()
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Config(BaseSettings):
    """Enthält die Konfiguration der App

    Die Konfiguration kann entweder über Umgebungsvariablen oder über eine .env-Datei bereitgestellt werden.
    Die .env-Datei sollte im Root-Verzeichnis der App liegen und den Namen "config.env" tragen.
    """
    envfile: str = "config.env"
    model_config = SettingsConfigDict(env_file=envfile, 
                    env_file_encoding="utf-8",
                    env_prefix="IND_",
                    case_sensitive=False)
    pg_host: str = Field(default="localhost")
    pg_port: str = Field(default="5432")
    pg_user: str = Field(default="postgres")
    pg_passwd: str = Field(default="postgres")
    pg_database: str = Field(default="postgres")
    pg_schema: str = Field(default="public")
    pg_api_key: str = Field(default="pg-docker")
    tika_noocr_url: str = Field(default="http://localhost:9998")
    tika_ocrurl: str = Field(default="http://localhost:9997")
    datadir: str = Field(default="./data")
    #docroot: str = Field(default="/home/rslsync/folders/")
    docsuf: Set[str] = Field(default={".pdf",".txt",".md",".docx",".doc",
        ".html",".htm",".epub",".odt",})
    picsuf: Set[str] = Field(default={".jpg", ".jpeg", ".bmp", ".png", ".heic", ".tiff", ".tif"})
    vidsuf: Set[str] = Field(default={".mov", ".vob", ".mkv", ".avi", ".mp4", ".mcf"})
    logger_name: str = "sqlalchemy.engine"
    loglevel: int = logging.INFO
    logger: logging.Logger = logging.Logger(logger_name, loglevel)
    
     
    
    def renew(self, **kwargs):
        """Erneuert die Konfiguration, z.B. nach Änderung der .env-Datei oder der Umgebungsvariablen"""
        self.__init__(**kwargs)
        
    def db_passwd_from_keyring(self) -> str|None:
        """DB Password aus Keyring holen, oder Default-Wert verwenden"""
        keyring.set_keyring(SecretServiceKeyring())
        pwd = keyring.get_password("API-Keys", self.pg_api_key)
        if pwd is None:
            self.logger.warning(f"Kein Passwort für API-Key '{self.pg_api_key}' im Keyring gefunden, verwende Default-Wert")
        else:
            self.pg_passwd = pwd
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
        database=self.pg_database,
)
            

    def get_ftype(self, suf: str) -> str:
        """Gibt den konfigurierten Dateityp zurück, basierend auf der Dateiendung"""
        _suf = suf.lower()
        if _suf in self.docsuf:
            return "doc"
        elif _suf in self.picsuf:
            return "pic"
        elif _suf in self.vidsuf:
            return "vid"
        else:
            return "unk"
            
