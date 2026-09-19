"""Componente DataLoader del framework de análisis crediticio.

Expone `DataLoader` como única clase pública para cargar archivos de datos
CSV, JSON o XLSX (ver `specs/001-dataloader-multiformat/`).
"""

from exceptions import (
    DataLoaderError,
    DataParsingError,
    FileNotFoundInSourceError,
    MissingFilePathError,
    UnsupportedFileFormatError,
)
from .loader import DataLoader

__all__ = [
    "DataLoader",
    "DataLoaderError",
    "DataParsingError",
    "FileNotFoundInSourceError",
    "MissingFilePathError",
    "UnsupportedFileFormatError",
]
