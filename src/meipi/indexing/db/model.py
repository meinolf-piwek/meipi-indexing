"""PostgreSQL database model for documents."""

# pylint: disable=E1136,W0105
import io
from typing import Optional, Self
from PIL import Image
import numpy as np
from imagehash import phash


from sqlalchemy import(    
    Index,
    MetaData,
    types,
    ForeignKey, 
    TEXT, DateTime,
    select
    )
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
    """

    impl = BYTEA

    cache_ok = True

    def process_bind_param(self, value: np.ndarray, dialect):
        bf = io.BytesIO()
        np.save(bf,value, allow_pickle=False)
        return bf.getvalue()

    def process_result_value(self, value, dialect):
        if value is not None:
            bf = io.BytesIO(value)
            return np.load(bf)
        else:
            return None
        #return np.array(value,dtype=np.uint8).reshape(224,224,3)
    
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
    pool: Mapped[str] = mapped_column(nullable=False,
                        doc = "Anwendungsgebiet, Datenpool, frei definierbar")
    path: Mapped[str] = mapped_column(nullable = False, unique=True, 
                        doc = "Pfad zur Datei, relativ zu einem root-Verzeichnis")
    fname: Mapped[str] = mapped_column(nullable=False,
                        doc="Dateiname")
    suffix: Mapped[str] = mapped_column(nullable=False,
                        doc="Dateisuffix, incl. dot")
    sort_date: Mapped[str] = mapped_column(DateTime(), nullable=False,
                            doc="Datum für Sortierung")
    fdate: Mapped[str] = mapped_column(DateTime(), nullable=False,
                        doc="Dateidatum des Systems")
    fsize: Mapped[int] = mapped_column(nullable=False,
                        doc="Dateigröße des Systems")
    clength: Mapped[int] = mapped_column(nullable=False,
                        doc="Content-length, aus Metadaten")
    ctype: Mapped[str] = mapped_column(nullable=False,
                        doc="Content type aus metadaten")
    md_keys: Mapped[Optional[list[str]]] = mapped_column(JSONB, nullable=True,
                        doc="Schlüssel der Metadaten")
    meta_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True,
                        doc="Metadaten als dictionary")
    sha256: Mapped[Optional[bytes]] = mapped_column(BYTEA, nullable=True, default=None,
                        doc="FileHash")
    
class DBMeta(Base, DBMetaMixin):
    """SQLAlchemy model for Meta data stored in PostgreSQL."""

    __tablename__ = "filemeta"
    __table_args__ = (
        Index("ix_filemeta_sha256", "sha256"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, sort_order=0,default=None)
    
    
class DBDoc(Base, DBMetaMixin):
    """SQLAlchemy model for documents stored in PostgreSQL."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, sort_order=0, default=None)
    inhalt: Mapped[str] = mapped_column(TEXT, default="", nullable=False, kw_only=True, deferred=True,
                        doc="Inhalt des Textdokuments")
    ts_content = mapped_column(
        TSVECTOR, Computed("to_tsvector('german', left(fname||inhalt,800000))"),
        deferred=True,
        doc="Spezieller Indexvektor für Volltextsuche in postgresql"
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
    def vector(cls) -> Mapped[list[float]]:  # pylint: disable=no-self-argument, missing-function-docstring
        return mapped_column(Vector(cls._vector_size), nullable=False)

    @declared_attr
    def doc(cls) -> Mapped[DBDoc]:  # pylint: disable=no-self-argument, missing-function-docstring
        return relationship()

    
class DBPic(Base, DBMetaMixin):
    """SQLAlchemy model for Pictures stored in PostgreSQL."""


    __tablename__ = "pictures"
    __table_args__ = (
        Index("ix_pictures_phash", "phash"),
    )
    _phash_size = 8
    _phash_high_freq = 2

    id: Mapped[int] = mapped_column(primary_key=True,  autoincrement=True, sort_order=0, default=None)
    xmp: Mapped[Optional[dict]] = mapped_column(JSONB, default=None,
                doc="XMP-attributes of the image")
    truncated: Mapped[Optional[bool]] = mapped_column(default=None,
                doc="Whether original image is truncated")
    thumbarray: Mapped[Optional[np.ndarray]] = mapped_column(PILArray,nullable=True,default=None,
                doc="Thumbnail 224x224x3 as ndarray")
    phash: Mapped[Optional[bytes]] = mapped_column(BYTEA,default=None,
                doc="Perceptual hash as bytes")
    
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
    def calc_phash(cls,im: Image)-> bytes:
        h = phash(im,cls._phash_size,cls._phash_high_freq)
        return bytes.fromhex(str(h))    