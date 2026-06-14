from __future__ import annotations

import re
import unicodedata

import config


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_EMOTICON_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:"
    r"[xX]?[DPOoT][_-]?[xX]?"
    r"|[:=;8xX][\-o\*']?[\)\]\(\[dDpP/:\}\{@\|\\]"
    r"|[\(\[]?[>＜]?[ＴTtOoQq；;][＿_\-\.]?[ＴTtOoQq；;][<＞]?[\)\]]?"
    r"|[\(\[]?[一-龥A-Za-z]?[·\._\-]?[一-龥A-Za-z]?[\)\]]"
    r")"
    r"(?![A-Za-z0-9])"
)
_BRACKETED_ASIDE_RE = re.compile(r"[\(（][^()（）]{1,24}[\)）]")
_LEADING_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-*+•]+|\d+[.)、])\s+")
_MARKDOWN_DECORATION_RE = re.compile(r"[*_#>|]+")
_REPEATED_PUNCT_RE = re.compile(r"([。！？!?；;，,、])\1+")
_DOT_RUN_RE = re.compile(r"(?:\.{2,}|…+)")
_DASH_RUN_RE = re.compile(r"[-‐‑‒–—―]{2,}")
_TILDE_RE = re.compile(r"[~～]+")
_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINE_RE = re.compile(r"\n{2,}")
_LAUGHTER_RE = re.compile(r"(?i)\b(?:lol|lmao|rofl|haha+|x+d+)\b|哈{2,}|呵{2,}|嘿{2,}|233+|www+")


def normalize_for_tts(text: str) -> str:
    if not config.TTS_TEXT_NORMALIZATION:
        return text.strip()

    cleaned = str(text)
    cleaned = _FENCED_CODE_RE.sub(config.TTS_CODE_REPLACEMENT, cleaned)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _URL_RE.sub(config.TTS_URL_REPLACEMENT, cleaned)
    cleaned = _INLINE_CODE_RE.sub(r"\1", cleaned)
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    cleaned = _LEADING_LIST_MARKER_RE.sub("", cleaned)
    cleaned = _MARKDOWN_DECORATION_RE.sub("", cleaned)
    cleaned = _LAUGHTER_RE.sub("", cleaned)
    cleaned = _strip_symbols(cleaned)
    cleaned = _BRACKETED_ASIDE_RE.sub("", cleaned)
    cleaned = _DOT_RUN_RE.sub("。", cleaned)
    cleaned = _DASH_RUN_RE.sub("，", cleaned)
    cleaned = _TILDE_RE.sub("", cleaned)
    cleaned = _REPEATED_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _SPACE_RE.sub(" ", cleaned)
    cleaned = _BLANK_LINE_RE.sub("\n", cleaned)
    cleaned = _tidy_punctuation(cleaned)
    return cleaned.strip(" \n，,。")


def _strip_symbols(text: str) -> str:
    chars: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category == "So":
            continue
        if category in {"Sk", "Cs", "Co"}:
            continue
        chars.append(char)
    return "".join(chars)


def _tidy_punctuation(text: str) -> str:
    text = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", text)
    text = re.sub(r"([，。！？；：、,.!?;:])\s+", r"\1", text)
    text = re.sub(r"[，,、]\s*([。！？!?；;])", r"\1", text)
    text = re.sub(r"([。！？!?；;])\s*[，,、]", r"\1", text)
    return text
