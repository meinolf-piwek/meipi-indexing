"""Clean Tika-extracted document text before storage and chunking."""

from __future__ import annotations

import re

_STYLE_SCRIPT = re.compile(
    r"<(?:style|script)\b[^>]*>.*?</(?:style|script)>",
    re.IGNORECASE | re.DOTALL,
)
_AT_RULE_BLOCK = re.compile(
    r"@(?!import\b)(?:font-face|media|keyframes|page|supports)\b[^{]*\{[^{}]*\}",
    re.IGNORECASE | re.DOTALL,
)
_CSS_RULE_BLOCK = re.compile(
    r"(?<![\w-])[.#][^{]+\{[^{}]*\}",
    re.DOTALL,
)
_CSS_DECLARATION = re.compile(
    r"\b(?:"
    r"font(?:-size|-family|-weight|-style|-variant|-stretch)?"
    r"|line-height|letter-spacing|word-spacing|text-(?:align|decoration|transform|indent|shadow)"
    r"|vertical-align|white-space|word-wrap|word-break"
    r"|color|background(?:-color|-image|-position|-repeat|-size)?"
    r"|border(?:-[a-z]+)?|outline(?:-[a-z]+)?"
    r"|margin(?:-[a-z]+)?|padding(?:-[a-z]+)?"
    r"|width|height|min-(?:width|height)|max-(?:width|height)"
    r"|display|position|float|clear|overflow(?:-[xy])?|visibility|opacity|z-index"
    r"|top|right|bottom|left"
    r")\s*:\s*[^;{}\n]+;?",
    re.IGNORECASE,
)
_EPUB_FONT_PATH = re.compile(
    r"\bOEBPS/Fonts/\S+\.(?:otf|ttf|woff2?|eot)\b",
    re.IGNORECASE,
)
_RTF_FONT = re.compile(r"\\(?:fs|f|cf|cb|pard|par)\d*\b")
_PDF_FONT_OP = re.compile(
    r"/(?:F\d+|Helvetica(?:-Bold|-Oblique)?|Times(?:-Bold|-Italic)?|Courier)\s+\d+(?:\.\d+)?\s+Tf\b"
)
_JS_FONTSIZE = re.compile(r"\b(?:this\.)?fontSize\s*[:=]\s*[\d.]+\b")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_JS_MARKER = re.compile(
    r"\b(?:var|let|const|function|return|if|else|for|while|switch|case|break|continue|"
    r"try|catch|throw|new|typeof|instanceof|void|delete|export|import|class|extends|async|await)\b|"
    r"=>|function\s*\(|__webpack_require__|window\.|document\.|\.prototype\.|module\.exports|"
    r"typeof\s+exports",
    re.IGNORECASE,
)
_CODE_ONLY_LINE = re.compile(r"^[\s\}\)\];,]+$")
_CODE_SYMBOLS = frozenset("{}();=<>[]&|!+*%/\\^~`@#$")


def _code_symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for char in text if char in _CODE_SYMBOLS) / len(text)


def _letter_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for char in text if char.isalpha()) / len(text)


def _looks_like_code_segment(text: str) -> bool:
    segment = text.strip()
    if not segment:
        return False
    if _letter_ratio(segment) > 0.55 and not _JS_MARKER.search(segment):
        return False
    if _JS_MARKER.search(segment) and _code_symbol_ratio(segment) > 0.06:
        return True
    if _code_symbol_ratio(segment) > 0.18:
        return True
    return (
        len(segment) > 200
        and _code_symbol_ratio(segment) > 0.12
        and segment.count(" ") / len(segment) < 0.12
    )


def strip_formatting_artifacts(text: str) -> str:
    """Remove font/CSS/HTML formatting noise often present in Tika-extracted text."""
    text = _STYLE_SCRIPT.sub(" ", text)
    for _ in range(3):
        updated = _AT_RULE_BLOCK.sub(" ", text)
        if updated == text:
            break
        text = updated
    for _ in range(5):
        updated = _CSS_RULE_BLOCK.sub(" ", text)
        if updated == text:
            break
        text = updated
    text = _CSS_DECLARATION.sub(" ", text)
    text = _EPUB_FONT_PATH.sub(" ", text)
    text = _RTF_FONT.sub(" ", text)
    text = _PDF_FONT_OP.sub(" ", text)
    text = _JS_FONTSIZE.sub(" ", text)
    return text


def _split_line_segments(line: str) -> list[str]:
    parts = re.split(r"(?<=[;}])\s*(?=[A-ZÄÖÜ„\"])", line)
    if len(parts) == 1:
        return [line]
    return parts


def strip_script_artifacts(text: str) -> str:
    """Remove JavaScript comment blocks and code-like lines from extracted HTML text."""
    text = _BLOCK_COMMENT.sub(" ", text)
    lines: list[str] = []
    for line in text.split("\n"):
        kept_parts: list[str] = []
        for segment in _split_line_segments(line):
            stripped = segment.strip()
            if not stripped:
                continue
            if stripped.startswith("//") and not stripped.startswith("///"):
                continue
            if _CODE_ONLY_LINE.match(stripped):
                continue
            if _looks_like_code_segment(stripped):
                continue
            kept_parts.append(stripped)
        if kept_parts:
            lines.append(" ".join(kept_parts))
        elif not line.strip():
            lines.append("")

    paragraphs: list[str] = []
    for para in re.split(r"\n{2,}", "\n".join(lines)):
        stripped = para.strip()
        if stripped and not _looks_like_code_segment(stripped):
            paragraphs.append(stripped)
    return "\n\n".join(paragraphs)


def strip_extraction_artifacts(text: str) -> str:
    """Remove formatting and script noise from Tika-extracted document text."""
    return strip_script_artifacts(strip_formatting_artifacts(text))


def normalize_document_paragraphs(text: str) -> str:
    """Collapse whitespace within paragraphs while preserving blank-line breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n{2,}", text)
    cleaned: list[str] = []
    for para in paragraphs:
        lines = [line.strip() for line in para.split("\n") if line.strip()]
        if not lines:
            continue
        cleaned.append(re.sub(r"[ \t]+", " ", " ".join(lines)))
    return "\n\n".join(cleaned).strip()


def clean_document_text(text: str) -> str:
    """Prepare Tika-extracted text for storage in ``DBMeta.inhalt``."""
    if not text:
        return ""
    return normalize_document_paragraphs(strip_extraction_artifacts(text))
