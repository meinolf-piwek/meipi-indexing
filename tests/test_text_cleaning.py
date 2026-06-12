"""Tests for document text cleaning."""

from meipi.indexing.text_cleaning import (
    clean_document_text,
    strip_formatting_artifacts,
    strip_script_artifacts,
)


def test_clean_text_preserves_paragraph_breaks() -> None:
    text = "First   paragraph.\n\nSecond\tparagraph.\nwith wrap"
    cleaned = clean_document_text(text)
    assert cleaned == "First paragraph.\n\nSecond paragraph. with wrap"


def test_split_paragraphs_after_cleaning() -> None:
    text = "Alpha line.\n\nBeta   line.\n\n\nGamma line."
    cleaned = clean_document_text(text)
    paragraphs = [p for p in cleaned.split("\n\n") if p]
    assert paragraphs == ["Alpha line.", "Beta line.", "Gamma line."]


def test_strip_formatting_artifacts_removes_css_and_font_paths() -> None:
    text = (
        ".navi { font-size: 11px; font-family: Arial; font-weight: bold; }\n\n"
        "Echter Absatz mit Inhalt.\n\n"
        "OEBPS/Fonts/SabonLTStd-Roman.otf"
    )
    cleaned = strip_formatting_artifacts(text)
    assert "font-size" not in cleaned
    assert "font-family" not in cleaned
    assert "OEBPS/Fonts/" not in cleaned
    assert "Echter Absatz mit Inhalt." in cleaned


def test_strip_formatting_artifacts_removes_minified_css_declarations() -> None:
    text = (
        "a, verdana, arial, sans-serif;font-size:11px;font-weight:normal;"
        "Actual sentence about contracts."
    )
    cleaned = strip_formatting_artifacts(text)
    assert "font-size" not in cleaned
    assert "font-weight" not in cleaned
    assert "Actual sentence about contracts." in cleaned


def test_strip_formatting_artifacts_preserves_words_containing_font() -> None:
    text = "Neben dem klapprigen Telefontischchen stand sie."
    assert strip_formatting_artifacts(text) == text


def test_strip_script_artifacts_removes_js_but_keeps_prose() -> None:
    text = (
        "/**\n * Copyright (c) Facebook, Inc.\n */\n"
        "try {(window.FB && !window.FB.__buffer) || (function(window) {\n"
        "  var apply = Function.prototype.apply;\n"
        "})();} catch (e) {}\n\n"
        "Der Vertrag wurde am Montag unterzeichnet.\n\n"
        "$:/config/OfficialPluginLibrary"
    )
    cleaned = strip_script_artifacts(text)
    assert "function(" not in cleaned
    assert "window.FB" not in cleaned
    assert "Der Vertrag wurde am Montag unterzeichnet." in cleaned
    assert "$:/config/OfficialPluginLibrary" in cleaned


def test_strip_script_artifacts_removes_minified_js_blob() -> None:
    text = (
        "var a=1,b=2;function x(){return a+b;}exports.foo=function(){return x();};"
        "Normaler Fließtext über Wirecard und Prüfberichte."
    )
    cleaned = strip_script_artifacts(text)
    assert "function(" not in cleaned
    assert "exports.foo" not in cleaned
    assert "Normaler Fließtext über Wirecard und Prüfberichte." in cleaned
