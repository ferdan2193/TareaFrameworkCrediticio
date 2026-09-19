"""Punto de entrada público del componente DataLoader.

`DataLoader` es la única clase que un desarrollador que consuma el
framework necesita conocer para cargar datos (ver
`specs/001-dataloader-multiformat/contracts/dataloader-api.md`). No expone
ninguna interfaz visual ni de consola: es una API puramente programática.
"""

import os

from exceptions import (
    FileNotFoundInSourceError,
    MissingFilePathError,
    UnsupportedFileFormatError,
)
from .readers import BaseFileReader, CsvReader, JsonReader, XlsxReader

import pandas as pd


class DataLoader:
    """Único punto de entrada para cargar archivos de datos CSV, JSON o XLSX.

    El formato se determina automáticamente a partir de la extensión del
    archivo indicado en `path`; no es necesario invocar un método distinto
    por formato (FR-005).
    """

    _READERS_POR_FORMATO: dict[str, type[BaseFileReader]] = {
        "csv": CsvReader,
        "json": JsonReader,
        "xlsx": XlsxReader,
    }

    def load(self, path: str) -> pd.DataFrame:
        """Carga el archivo en `path` y devuelve su contenido como DataFrame.

        Args:
            path: Ruta al archivo `.csv`, `.json` o `.xlsx` a cargar.
                Debe proporcionarse en cada llamada; no tiene valor por
                defecto (FR-001).

        Returns:
            Un `pandas.DataFrame` con los datos del archivo. Puede tener 0
            filas si el archivo de origen está vacío, pero nunca es `None`
            (FR-006).

        Raises:
            MissingFilePathError: si `path` es `None` o una cadena vacía.
            FileNotFoundInSourceError: si `path` no existe en el sistema de
                archivos.
            UnsupportedFileFormatError: si la extensión de `path` no es
                `.csv`, `.json` ni `.xlsx`.
            DataParsingError: si el archivo tiene un formato soportado pero
                su contenido no puede interpretarse.
        """
        if not path:
            raise MissingFilePathError("Se requiere una ruta de archivo para cargar los datos.")

        if not os.path.isfile(path):
            raise FileNotFoundInSourceError(f"No se encontró el archivo '{path}'.")

        formato = os.path.splitext(path)[1].lstrip(".").lower()
        reader_cls = self._READERS_POR_FORMATO.get(formato)
        if reader_cls is None:
            soportados = ", ".join(sorted(self._READERS_POR_FORMATO))
            raise UnsupportedFileFormatError(
                f"Formato '.{formato}' no soportado para '{path}'. Formatos soportados: {soportados}."
            )

        reader: BaseFileReader = reader_cls()
        return reader.read(path)
