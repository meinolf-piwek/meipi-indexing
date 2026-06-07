"""Thumbnail generation for non-image document files."""

from __future__ import annotations

import textwrap
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .thumbnail import THUMB_SIZE, thumbnail_from_image

_TEXT_SUFFIXES = {".txt", ".md", ".csv"}
_HTML_SUFFIXES = {".html", ".htm"}
_TEXT_PREVIEW_CHARS = 600
_TEXT_LINE_WIDTH = 28
_TEXT_MAX_LINES = 10


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self._parts)


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_preview_text(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return ["(empty)"]
    wrapped = textwrap.wrap(normalized, width=_TEXT_LINE_WIDTH)
    return wrapped[:_TEXT_MAX_LINES] or ["(empty)"]


def _thumbnail_from_text(
    text: str,
    *,
    title: str | None = None,
    size: int = THUMB_SIZE,
) -> np.ndarray:
    image = Image.new("RGB", (size, size), color=(248, 248, 248))
    draw = ImageDraw.Draw(image)
    title_font = _load_font(16)
    body_font = _load_font(14)

    y = 12
    if title:
        draw.text((12, y), title, fill=(64, 64, 64), font=title_font)
        y += 22

    for line in _wrap_preview_text(text):
        draw.text((12, y), line, fill=(32, 32, 32), font=body_font)
        y += 18
        if y > size - 12:
            break

    return np.asarray(image)


def _thumbnail_placeholder(
    label: str,
    *,
    subtitle: str | None = None,
    size: int = THUMB_SIZE,
) -> np.ndarray:
    image = Image.new("RGB", (size, size), color=(232, 236, 240))
    draw = ImageDraw.Draw(image)
    font = _load_font(18)
    small_font = _load_font(14)

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = max(8, (size - text_width) // 2)
    y = (size - text_height) // 2 - (10 if subtitle else 0)
    draw.text((x, y), label, fill=(48, 56, 64), font=font)

    if subtitle:
        sub_bbox = draw.textbbox((0, 0), subtitle, font=small_font)
        sub_width = sub_bbox[2] - sub_bbox[0]
        sub_x = max(8, (size - sub_width) // 2)
        draw.text((sub_x, y + text_height + 8), subtitle, fill=(96, 104, 112), font=small_font)

    return np.asarray(image)


def _read_text_preview(filepath: str) -> str:
    with open(filepath, encoding="utf-8", errors="replace") as handle:
        return handle.read(_TEXT_PREVIEW_CHARS)


def _thumbnail_from_text_file(filepath: str, size: int = THUMB_SIZE) -> np.ndarray:
    return _thumbnail_from_text(_read_text_preview(filepath), size=size)


def _thumbnail_from_html_file(filepath: str, size: int = THUMB_SIZE) -> np.ndarray:
    parser = _HTMLTextExtractor()
    parser.feed(_read_text_preview(filepath))
    return _thumbnail_from_text(parser.text(), size=size)


def _thumbnail_from_pdf(filepath: str, size: int = THUMB_SIZE) -> np.ndarray:
    import pymupdf

    with pymupdf.open(filepath) as document:
        if document.page_count == 0:
            return _thumbnail_placeholder(
                Path(filepath).suffix.upper() or ".PDF",
                subtitle=Path(filepath).name,
                size=size,
            )
        page = document.load_page(0)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return thumbnail_from_image(image, size)


def make_document_thumbnail(filepath: str, size: int = THUMB_SIZE) -> np.ndarray:
    """Create a preview thumbnail for a document file.

    Supports rendered previews for PDF and text-like formats. Other configured
    document suffixes fall back to a simple labeled placeholder image.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _thumbnail_from_pdf(filepath, size)
    if suffix in _TEXT_SUFFIXES:
        return _thumbnail_from_text_file(filepath, size)
    if suffix in _HTML_SUFFIXES:
        return _thumbnail_from_html_file(filepath, size)

    return _thumbnail_placeholder(
        suffix.upper() or "DOC",
        subtitle=path.name,
        size=size,
    )
