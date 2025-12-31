"""Top-level package for Story For You."""
from importlib import metadata

try:
    __version__ = metadata.version("story-for-you")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
