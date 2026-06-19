"""PostgreSQL database model for pictures, documents and their metadata.
Es wird SQLAlchemy ORM verwendet, um die Datenbanktabellen zu definieren und zu verwalten.
Die Modelle umfassen

    * :class:`DBMeta`: Tabelle für Meta-Daten von Dateien (inkl. Volltext ``inhalt``)
    * :class:`DBDoc`: Optionale Zeile pro Dokument-Datei (z. B. für Embedding-Chunks)
    * :class:`DBPic`: Tabelle für Bilder mit Thumbnail und Perceptual Hash
    * :class:`DBDinoV2Vector`: Tabelle für DINO V2 Bildvektoren
    
Die Mixins :class:`DBMetaMixin`, :class:`DocVectorMixin` und :class:`PicVectorMixin` 
bieten gemeinsame Felder und Methoden für die jeweiligen Modelle. 

Die Modelle enthalten Methoden zum Erstellen und Löschen von Tabellen 
sowie zur Durchführung von Volltextsuchen und Berechnung von Perceptual Hashes."""
#TODO: Tabellen für Dokumenten- und Bildvektoren, die von Embedder-Modellen erstellt werden, hinzufügen

import io
#import types
#from tkinter import CASCADE
from typing import Optional, Self, Sequence, List, Tuple
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
from datetime import datetime
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
    CascadeOptions
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, BYTEA, BIGINT
from sqlalchemy.schema import Computed
from pgvector.sqlalchemy import Vector

_search_language = "german"

# Shared ORM metadata: tables stay unqualified; PostgreSQL schema comes from search_path (pg_schema).
orm_metadata = MetaData()


# (filesystem path, filemeta.id)
type IdList = Sequence[Tuple[str, int]]

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

    def process_bind_param(self, value: np.ndarray|None, dialect):
        if value is None:
            return None
        bf = io.BytesIO()
        np.save(bf, value, allow_pickle=False)
        return bf.getvalue()

    def process_result_value(self, value, dialect):
        if value is not None:
            bf = io.BytesIO(value)
            return np.load(bf, allow_pickle=False)
        else:
            return None
        

    def process_literal_param(self, value, dialect): #type: ignore
        return None 

    def coerce_compared_value(self, op, value):
        return self.impl.coerce_compared_value(op=op, value=value) #type: ignore


class Base(MappedAsDataclass, DeclarativeBase):
    """Base class for SQLAlchemy models."""

    metadata = orm_metadata

    @classmethod
    def create_table(cls, session: Session) -> None:
        """Create the table in the database."""
        if session.bind is None:
            raise ValueError(
                "Session is not bound to an engine. \
                Ensure the session is properly configured."
            )
        cls.metadata.create_all(session.bind, tables=[cls.__table__]) #type: ignore

    @classmethod
    def drop_table(cls, session: Session) -> None:
        """Drop the table from the database."""
        if session.bind is None:
            raise ValueError(
                "Session is not bound to an engine. \
                Ensure the session is properly configured."
            )
        cls.metadata.drop_all(session.bind, tables=[cls.__table__]) #type: ignore

    def as_dict(self):
        """Erzeugt Dictionary ohne _sa_instance_state"""
        data = self.__dict__.copy()
        data.pop("_sa_instance_state", "")  # Remove SQLAlchemy state
        return data


class CatalogBase(DeclarativeBase):
    """Declarative base for PostgreSQL catalog views (read-only, no dataclass ORM)."""

    metadata = MetaData()


class DBCatalog(CatalogBase):
    """Read-only ORM mapping of ``pg_catalog.pg_tables``."""

    __tablename__ = "pg_tables"
    __table_args__ = {"schema": "pg_catalog"}

    schemaname: Mapped[str] = mapped_column(primary_key=True)
    tablename: Mapped[str] = mapped_column(primary_key=True)
    tableowner: Mapped[str] = mapped_column()
    tablespace: Mapped[Optional[str]] = mapped_column()
    hasindexes: Mapped[bool] = mapped_column()
    hasrules: Mapped[bool] = mapped_column()
    hastriggers: Mapped[bool] = mapped_column()
    rowsecurity: Mapped[bool] = mapped_column()


class DBPool(Base):
    """SQLAlchemy model for data pools stored in PostgreSQL.
    
    Diese Tabelle dient zur Verwaltung von Datenpools, die als logische Gruppen von Dateien definiert werden können.
    Ein Datenpool könnte beispielsweise ein bestimmtes Anwendungsgebiet oder eine Kategorie von Dateien repräsentieren
    """

    __tablename__ = "datapools"
    __table_args__ = (Index("ix_datapools_pool", "pool", unique=True),)

    
    pool: Mapped[str] = mapped_column(
        nullable=False, unique=True, doc="Name of the data pool"
    )
    description: Mapped[Optional[str]] = mapped_column(
        nullable=True, doc="Optional description of the data pool"
    )
    id: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True, sort_order=0, default=None
    )

class DBMeta(Base):
    """SQLAlchemy model for Meta data stored in PostgreSQL."""

    __tablename__ = "filemeta"
    __table_args__ = (
        Index("ix_filemeta_sha256", "sha256"),
        Index("ix_filemeta_path", "pool_id", "path", unique=True),
        Index("ix_filemeta_ts_content", "ts_content", postgresql_using="gin"),
        Index("ix_filemeta_phash", "phash"),
    )
    _phash_size = 8
    _phash_high_freq = 2

    id: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True, sort_order=0, init=False
            )
    pool_id: Mapped[int] = mapped_column(ForeignKey("datapools.id"),
        nullable=False, doc="Anwendungsgebiet, Datenpool, frei definierbar"
    )
    path: Mapped[str] = mapped_column(
        nullable=False,
        #unique=True,
        doc="Pfad zur Datei, relativ zu einem root-Verzeichnis",
    )
    fname: Mapped[str] = mapped_column(nullable=False, doc="Dateiname")
    suffix: Mapped[str] = mapped_column(nullable=False, doc="Dateisuffix, incl. dot")
    sort_date: Mapped[datetime] = mapped_column(
        DateTime(), nullable=False, doc="Datum für Sortierung"
    )
    fdate: Mapped[datetime] = mapped_column(
        DateTime(), nullable=False, doc="Dateidatum des Systems"
    )
    fsize: Mapped[int] = mapped_column(BIGINT, nullable=False, doc="Dateigröße des Systems")
    clength: Mapped[int] = mapped_column( BIGINT,
        nullable=False, doc="Content-length, aus Metadaten"
    )
    ctype: Mapped[str] = mapped_column(nullable=False,
                        doc="Content type aus metadaten")
    ftype: Mapped[str] = mapped_column(nullable=False,
                        doc="Dateityp, konfiguriert in config.py")
    md_keys: Mapped[Optional[list[str]]] = mapped_column(
        JSONB, nullable=True, doc="Schlüssel der Metadaten"
    )
    meta_data: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True, doc="Metadaten als dictionary"
    )
    sha256: Mapped[Optional[bytes]] = mapped_column(
        BYTEA, nullable=True, default=None, doc="FileHash"
    )
    thumbarray: Mapped[Optional[np.ndarray]] = mapped_column(
        PILArray, nullable=True, default=None, doc="Thumbnail 224x224x3 as ndarray"
    )
    phash: Mapped[Optional[bytes]] = mapped_column(
        BYTEA, default=None, doc="Perceptual hash of the thumbnail as bytes"
    )
    inhalt: Mapped[str] = mapped_column(
        TEXT,
        default="",
        nullable=False,
        kw_only=True,
        doc="Extracted text content (all file types)",
    )
    ts_content: Mapped[TSVECTOR] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('%s', inhalt)" % _search_language),
        init=False,
        nullable=True,
        deferred=True,
        doc="Full-text search vector derived from inhalt",
    )
    # doc: Mapped[Optional["DBDoc"]] = relationship(back_populates="meta",
    #                                 cascade = "all, delete-orphan", passive_deletes=True, default= None)
    pic: Mapped[Optional["DBPic"]] = relationship(back_populates="meta",
                                    cascade = "all, delete-orphan", passive_deletes=True, default= None)
    vid: Mapped[Optional["DBVid"]] = relationship(back_populates="meta",
                                    cascade = "all, delete-orphan", passive_deletes=True, default= None)

    @property
    def thumb(self) -> Image.Image | None:
        if self.thumbarray is not None:
            return Image.fromarray(self.thumbarray)
        return None

    def set_phash(self) -> None:
        if self.thumbarray is not None:
            self.phash = self.calc_phash(Image.fromarray(self.thumbarray))

    @classmethod
    def calc_phash(cls, im: Image.Image) -> bytes:
        h = phash(im, cls._phash_size, cls._phash_high_freq)
        return bytes.fromhex(str(h))

    @classmethod
    def tsquery(
        cls: type[Self], query: str, session: Session, lang: str = "german"
    ) -> Sequence[Self]:
        """Perform a full-text search on ``ts_content``."""
        stmt = select(cls).where(cls.ts_content.match(query, reg_conf=lang))
        return session.execute(stmt).scalars().all()


# class DBDoc(Base):
#     """Optional document row for ``doc`` filemeta (e.g. embedding chunks).

#     Full-text content lives on :class:`DBMeta` (``inhalt`` / ``ts_content``).
#     """

#     __tablename__ = "documents"

#     id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, sort_order=0, init=False)

#     meta_id: Mapped[int] = mapped_column(
#         ForeignKey("filemeta.id", ondelete="CASCADE"),
#         nullable=False,
#         unique=True,
#         init=False,
#     )
#     meta: Mapped["DBMeta"] = relationship(
#         back_populates="doc", init=False, single_parent=True
#     )
#     chunks: Mapped[list["DBBgeM3Vector"]] = relationship(
#         back_populates="doc",
#         default_factory=list,
#         cascade="all, delete-orphan",
#         passive_deletes=True,
#     )


class DBPic(Base):
    """SQLAlchemy model for Picture data stored in PostgreSQL.

    Enthält XMP-Metadaten. Thumbnail und Perceptual Hash liegen auf :class:`DBMeta`.
    """

    __tablename__ = "pictures"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, sort_order=0, init=False)
    # Fremdschlüssel auf die Tabelle der Dateimetadaten
    meta_id: Mapped[int] = mapped_column(ForeignKey("filemeta.id",ondelete="CASCADE"), nullable=False, unique=True, init=False)
    meta: Mapped["DBMeta"] = relationship(back_populates="pic", init=False, single_parent=True)

    xmp: Mapped[Optional[dict]] = mapped_column(
        JSONB, default=None, doc="XMP-attributes of the image"
    )
    truncated: Mapped[Optional[bool]] = mapped_column(
        default=None, doc="Whether original image is truncated"
    )


class DBVid(Base):
    """SQLAlchemy model for Video data stored in PostgreSQL."""
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, sort_order=0,init=False)
    # Fremdschlüssel auf die Tabelle der Dateimetadaten
    meta_id: Mapped[int] = mapped_column(ForeignKey("filemeta.id",ondelete="CASCADE"),
        nullable=False, unique=True, init=False) 
    meta: Mapped["DBMeta"] = relationship(back_populates="vid",init=False, single_parent=True)


class ChunkItem(MappedAsDataclass):
    doc_id: Mapped[int] = mapped_column(ForeignKey("filemeta.id", ondelete="CASCADE"), primary_key=True)
    chunk_index: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(TEXT, nullable=False)

    

class DocVectorMixin(ChunkItem):
    """Mixin für DocVectorTables"""

    @classmethod
    @abstractmethod
    def _vector_size(cls) -> int:
        raise NotImplementedError("Subclasses must implement this method")

    @declared_attr
    def vector(
        cls,
    ) -> Mapped[list[float] | None]:
        return mapped_column(Vector(cls._vector_size()), nullable=True, default=None)

    @declared_attr
    def doc(
        cls,
    ) -> Mapped["DBMeta"]:
        return relationship(DBMeta, init=False, single_parent=True)


class PicVectorMixin(MappedAsDataclass):
    """Mixin für PicVectorTables
    
    Es enthält ein spezielles Feld vector, das die von einem Embedder-Modell erstellten Vektoren speichert.
    Die Größe des Vektors wird durch die Klasse definiert, die dieses Mixin verwendet.
    Es wird ein Fremdschlüssel pic_id definiert, der auf die Tabelle der Bilddaten verweist.
    """

    @classmethod
    @abstractmethod
    def _vector_size(cls) -> int:
        raise NotImplementedError("Subclasses must implement this method")

    pic_id: Mapped[int] = mapped_column(ForeignKey("pictures.id",ondelete="CASCADE"), primary_key=True)

    @declared_attr
    def vector(
        cls,
    ) -> Mapped[
        list[float]
    ]:
        return mapped_column(Vector(cls._vector_size()), nullable=False)


class DBDinoV2Vector(Base, PicVectorMixin):
    """SQLAlchemy model for DINO V2 image embeddings stored in PostgreSQL."""

    __tablename__ = "dino_v2_vectors"
    @classmethod
    def _vector_size(cls) -> int:
        return 1024  # Größe des Vektors für DINO-Modelle


class DBBgeM3Vector(Base, DocVectorMixin):
    """Chunk text and BGE-M3 embedding vectors for indexed documents."""

    __tablename__ = "bge_m3_vectors"

    @classmethod
    def _vector_size(cls) -> int:
        return 1024


def _unqualify_orm_tables() -> None:
    """Keep table names unqualified so ``pg_schema`` can change via search_path."""
    for mapper in Base.registry.mappers:
        mapper.persist_selectable.schema = None


_unqualify_orm_tables()
