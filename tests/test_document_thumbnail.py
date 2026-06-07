"""Tests for document thumbnail generation."""

from __future__ import annotations

import numpy as np
import pymupdf
from PIL import Image

from meipi.indexing.document_thumbnail import make_document_thumbnail
from meipi.indexing.thumbnail import THUMB_SIZE


def test_make_document_thumbnail_from_txt(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("Vertrag\nProjektplan\nRechnung", encoding="utf-8")

    thumb = make_document_thumbnail(str(path))

    assert thumb.shape == (THUMB_SIZE, THUMB_SIZE, 3)
    assert thumb.dtype == np.uint8
    assert thumb.mean() < 250


def test_make_document_thumbnail_from_html(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<html><body><p>Hello <b>world</b></p></body></html>", encoding="utf-8")

    thumb = make_document_thumbnail(str(path))

    assert thumb.shape == (THUMB_SIZE, THUMB_SIZE, 3)
    assert thumb.dtype == np.uint8


def test_make_document_thumbnail_from_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "PDF preview")
    document.save(path)
    document.close()

    thumb = make_document_thumbnail(str(path))

    assert thumb.shape == (THUMB_SIZE, THUMB_SIZE, 3)
    assert thumb.dtype == np.uint8
    assert thumb.std() > 0


def test_make_document_thumbnail_placeholder_for_docx(tmp_path):
    path = tmp_path / "report.docx"
    path.write_bytes(b"not a real docx")

    thumb = make_document_thumbnail(str(path))

    assert thumb.shape == (THUMB_SIZE, THUMB_SIZE, 3)
    assert thumb.dtype == np.uint8
    assert thumb[thumb.shape[0] // 2, thumb.shape[1] // 2].mean() < 240


def test_thumbnail_from_image_still_used_by_image_helper(tmp_path):
    from meipi.indexing.thumbnail import make_thumbnail_array

    path = tmp_path / "photo.png"
    Image.new("RGB", (300, 150), color=(10, 20, 30)).save(path)

    thumb = make_thumbnail_array(str(path))

    assert thumb.shape == (THUMB_SIZE, THUMB_SIZE, 3)
