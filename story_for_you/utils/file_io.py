import codecs
import hashlib
import logging
from collections.abc import Iterable
from pathlib import Path

try:  # pragma: no cover - dependency is declared but we stay defensive
    from charset_normalizer import from_bytes as _charset_from_bytes
except ImportError:  # pragma: no cover
    _charset_from_bytes = None


logger = logging.getLogger(__name__)

_ENCODING_FALLBACKS = ("gb18030", "gbk", "big5")


def _normalize_encoding_label(value: str | None) -> str | None:
    if not value:
        return None
    return value.lower().replace("_", "-")


def _unique_encodings(candidates: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        normalized = _normalize_encoding_label(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _encoding_from_bom(data: bytes) -> str | None:
    if data.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if data.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le"
    if data.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be"
    if data.startswith(codecs.BOM_UTF32_LE):
        return "utf-32-le"
    if data.startswith(codecs.BOM_UTF32_BE):
        return "utf-32-be"
    return None


def _detect_encoding(data: bytes) -> str | None:
    if not _charset_from_bytes:
        return None
    result = _charset_from_bytes(data).best()
    if not result or not result.encoding:
        return None
    if result.encoding.lower() == "ascii":
        return "utf-8"
    return _normalize_encoding_label(result.encoding)


def compute_file_hash(file_path: Path, length: int = 16) -> str:
    """Compute a truncated SHA-256 hex digest of the file content."""
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def read_text_file(file_path: Path, encoding: str | None = None) -> str:
    """Read a text file with graceful decoding fallbacks."""
    data = file_path.read_bytes()
    if not data:
        return ""

    preferred = _normalize_encoding_label(encoding)
    bom_encoding = _encoding_from_bom(data)
    detected = _detect_encoding(data)

    candidates = _unique_encodings(
        (
            preferred,
            "utf-8",
            "utf-8-sig",
            bom_encoding,
            detected,
            *_ENCODING_FALLBACKS,
        )
    )

    last_error: UnicodeDecodeError | None = None
    for candidate in candidates:
        try:
            text = data.decode(candidate)
            if candidate not in {"utf-8", preferred}:
                logger.debug("Decoded %s using encoding '%s'", file_path, candidate)
            return text
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error:
        raise last_error

    # As a last resort, fall back to UTF-8 replacement to avoid crashing the CLI.
    return data.decode("utf-8", errors="replace")


def write_text_file(file_path: Path, content: str) -> None:
    """Write UTF-8 content to disk, creating parents when needed."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
