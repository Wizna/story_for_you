from pathlib import Path


def read_text_file(file_path: Path) -> str:
    """Read a UTF-8 text file."""
    return file_path.read_text(encoding="utf-8")


def write_text_file(file_path: Path, content: str) -> None:
    """Write UTF-8 content to disk, creating parents when needed."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
