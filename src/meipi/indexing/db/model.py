"""PostgreSQL database model for pictures, documents and their metadata.
Es wird SQLAlchemy ORM verwendet, um die Datenbanktabellen zu definieren und zu verwalten.
Die Modelle umfassen:
- DBMeta: Tabelle für Meta-Daten von Dateien
- DBDoc: Tabelle für Textdokumente mit Volltextindex
- DBPic: Tabelle für Bilder mit Thumbnail und Perceptual Hash
- DBDinoV2Vector: Tabelle für DINO V2 Bildvektoren
Die Mixins DBMetaMixin, DocVectorMixin und PicVectorMixin 
bieten gemeinsame Felder und Methoden für die jeweiligen Modelle. 
Die Modelle enthalten Methoden zum Erstellen und Löschen von Tabellen 
sowie zur Durchführung von Volltextsuchen und Berechnung von Perceptual Hashes."""
#TODO: Tabellen für Dokumenten- und Bildvektoren, die von Embedder-Modellen erstellt werden, hinzufügen

import io
from typing import Optional, Self
from PIL import Image
import numpy as np
from imagehash import phash


from sqlalchemy import Index, MetaData, types, ForeignKey, TEXT, DateTime, select
from sqlalchemy.orm import (
    Mapped,
    DeclarativeBase,
    relationship,
    mapped_column,
    MappedAsDataclass,
    declared_attr,
    Session,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, BYTEA
from sqlalchemy.schema import Computed
from pgvector.sqlalchemy import Vector


class PILArray(types.TypeDecorator):
    """
    Type for PIL Image as numpy array
    
    Damit können Thumbnails als numpy arrays in der Datenbank gespeichert werden,
    ohne sie vorher in ein anderes Format konvertieren zu müssen.
    Der Datenbanktyp ist BYTEA, da die numpy arrays als Binärdaten gespeichert werden.
    """

    impl = BYTEA
    cache_ok = True

    @property
    def python_type(self) -> type[np.ndarray]:
        return np.ndarray

    def process_bind_param(self, value: np.ndarray, dialect):
        bf = io.BytesIO()
        np.save(bf, value, allow_pickle=False)
        return bf.getvalue()

    def process_result_value(self, value, dialect):
        if value is not None:
            bf = io.BytesIO(value)
            return np.load(bf)
        else:
            return None
        # return np.array(value,dtype=np.uint8).reshape(224,224,3)

    def process_literal_param(self, value, dialect):
        return self.process_bind_param(value, dialect)

    def coerce_compared_value(self, op, value):
        return self.impl.coerce_compared_value(op, value)


class Base(MappedAsDataclass, DeclarativeBase):
    """Base class for SQLAlchemy models."""

    metadata = MetaData("public")

    @classmethod
    def create_table(cls, session: Session) -> None:
        """Create the table in the database."""
        cls.metadata.create_all(session.bind, tables=[cls.__table__])

    @classmethod
    def drop_table(cls, session: Session) -> None:
        """Drop the table from the database."""
        cls.metadata.drop_all(session.bind, tables=[cls.__table__])

    def as_dict(self):
        """Erzeugt Dictionary ohne _sa_instance_state"""
        data = self.__dict__.copy()
        data.pop("_sa_instance_state", "")  # Remove SQLAlchemy state
        return data


class DBMetaMixin(MappedAsDataclass):
    """Mixin für Meta-Daten Felder"""

    pool: Mapped[str] = mapped_column(
        nullable=False, doc="Anwendungsgebiet, Datenpool, frei definierbar"
    )
    path: Mapped[str] = mapped_column(
        nullable=False,
        unique=True,
        doc="Pfad zur Datei, relativ zu einem root-Verzeichnis",
    )
    fname: Mapped[str] = mapped_column(nullable=False, doc="Dateiname")
    suffix: Mapped[str] = mapped_column(nullable=False, doc="Dateisuffix, incl. dot")
    sort_date: Mapped[str] = mapped_column(
        DateTime(), nullable=False, doc="Datum für Sortierung"
    )
    fdate: Mapped[str] = mapped_column(
        DateTime(), nullable=False, doc="Dateidatum des Systems"
    )
    fsize: Mapped[int] = mapped_column(nullable=False, doc="Dateigröße des Systems")
    clength: Mapped[int] = mapped_column(
        nullable=False, doc="Content-length, aus Metadaten"
    )
    ctype: Mapped[str] = mapped_column(nullable=False, doc="Content type aus metadaten")
    md_keys: Mapped[Optional[list[str]]] = mapped_column(
        JSONB, nullable=True, doc="Schlüssel der Metadaten"
    )
    meta_data: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True, doc="Metadaten als dictionary"
    )
    sha256: Mapped[Optional[bytes]] = mapped_column(
        BYTEA, nullable=True, default=None, doc="FileHash"
    )


class DBMeta(Base, DBMetaMixin):
    """SQLAlchemy model for Meta data stored in PostgreSQL."""

    __tablename__ = "filemeta"
    __table_args__ = (Index("ix_filemeta_sha256", "sha256"),)

    id: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True, sort_order=0, default=None
    )


class DBDoc(Base, DBMetaMixin):
    """SQLAlchemy model for text documents stored in PostgreSQL.
    
    Es enthält ein spezielles Feld ts_content, das als TSVECTOR definiert ist und eine
    Volltextsuche in PostgreSQL ermöglicht. 
    Die Methode tsquery ermöglicht es, eine PostgreSQL-Volltextsuche auf diesem Feld durchzuführen.
    """

    __tablename__ = "documents"
    _search_language = "german"

    id: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True, sort_order=0, default=None
    )
    inhalt: Mapped[str] = mapped_column(
        TEXT,
        default="",
        nullable=False,
        kw_only=True,
        deferred=True,
        doc="Inhalt des Textdokuments",
    )
    ts_content = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('%s', left(fname||inhalt,800000))" % _search_language),
        deferred=True,
        doc="Spezieller Indexvektor für Volltextsuche in postgresql",
    )

    @classmethod
    def tsquery(cls: Self, query: str, session: Session, lang="german") -> list[Self]:
        """Perform a full-text search on the ts_content field."""
        stmt = select(cls).where(cls.ts_content.match(query, reg_conf=lang))
        return session.execute(stmt).scalars().all()


class DocVectorMixin(MappedAsDataclass):
    """Mixin für DocVectorTables"""

    _vector_size = 0
    chunk_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    doc_id: Mapped[id] = mapped_column(ForeignKey("documents.id"))
    content: Mapped[str] = mapped_column(TEXT, nullable=True)

    @declared_attr
    def vector(
        cls,
    ) -> Mapped[
        list[float]
    ]:  
        return mapped_column(Vector(cls._vector_size), nullable=False)

    @declared_attr
    def doc(
        cls,
    ) -> Mapped[DBDoc]:
        return relationship()


class DBPic(Base, DBMetaMixin):
    """SQLAlchemy model for Picture data stored in PostgreSQL.
    
    Es enthält neben den Datei-Metadaten Felder für XMP-Metadaten, 
    ein Thumbnail als numpy array und einen Perceptual Hash.
    Die eigentlichen Bilddaten werden nicht in der Datenbank gespeichert, sondern nur die Metadaten und der Hash.
    Die Methode set_phash berechnet den Perceptual Hash basierend auf dem Thumbnail, falls dieses vorhanden ist. 
    Der Perceptual Hash wird als BYTEA gespeichert, um eine effiziente Speicherung und Suche zu ermöglichen. 
    Es wird ein Index auf dem phash-Feld erstellt, um schnelle Ähnlichkeitssuchen zu ermöglichen.
    """

    __tablename__ = "pictures"
    __table_args__ = (Index("ix_pictures_phash", "phash"),)
    _phash_size = 8
    _phash_high_freq = 2

    id: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True, sort_order=0, default=None
    )
    xmp: Mapped[Optional[dict]] = mapped_column(
        JSONB, default=None, doc="XMP-attributes of the image"
    )
    truncated: Mapped[Optional[bool]] = mapped_column(
        default=None, doc="Whether original image is truncated"
    )
    thumbarray: Mapped[Optional[np.ndarray]] = mapped_column(
        PILArray, nullable=True, default=None, doc="Thumbnail 224x224x3 as ndarray"
    )
    phash: Mapped[Optional[bytes]] = mapped_column(
        BYTEA, default=None, doc="Perceptual hash as bytes"
    )

    def set_phash(self):
        if self.thumbarray is not None:
            self.phash = self.calc_phash(self.thumb)

    @property
    def thumb(self):
        if self.thumbarray is not None:
            return Image.fromarray(self.thumbarray)
        else:
            return None

    @classmethod
    def calc_phash(cls, im: Image) -> bytes:
        h = phash(im, cls._phash_size, cls._phash_high_freq)
        return bytes.fromhex(str(h))


class PicVectorMixin(MappedAsDataclass):
    """Mixin für PicVectorTables
    
    Es enthält ein spezielles Feld vector, das die von einem Embedder-Modell erstellten Vektoren speichert.
    Die Größe des Vektors wird durch die Klasse definiert, die dieses Mixin verwendet.
    Es wird ein Fremdschlüssel pic_id definiert, der auf die Tabelle der Bilddaten verweist.
    """

    _vector_size = 0
    pic_id: Mapped[int] = mapped_column(ForeignKey("pictures.id"), primary_key=True)

    @declared_attr
    def vector(
        cls,
    ) -> Mapped[
        list[float]
    ]:
        return mapped_column(Vector(cls._vector_size), nullable=False)


class DBDinoV2Vector(Base, PicVectorMixin):
    """SQLAlchemy model for DINO V2 image embeddings stored in PostgreSQL."""

    __tablename__ = "dino_v2 vectors"
    _vector_size = 1024  # Größe des Vektors für DINO-Modelle
