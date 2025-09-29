"""Wird aktuell nicht genutzt - nur als Beispiel"""
# pylint: disable=E1136,W0105, abstract-method
import io
from sqlalchemy.types import TypeDecorator, LargeBinary
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)
from PIL import Image
from .model import Base

class PILImageType(TypeDecorator):
    """Decorator für Bilder-Attribut"""
    impl = LargeBinary
               
    def process_bind_param(self, value:Image.Image | None, dialect):
        if value is None:
            return None
        buf = io.BytesIO()
        value.save(buf, format="JPEG")
        return buf.getvalue()
    
    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return Image.open(io.BytesIO(value))
        
class Photo(Base):
    __tablename__ = "photos"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    image: Mapped[Image.Image] = mapped_column(PILImageType)