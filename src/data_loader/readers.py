"""Readers concretos por formato de archivo.

`BaseFileReader` define el contrato abstracto que toda estrategia de
lectura debe cumplir (pilar de Abstracción). Cada subclase concreta
encapsula los detalles de lectura de su formato y traduce errores de bajo
nivel a `DataParsingError` (pilar de Encapsulamiento), extendiendo la clase
base (pilar de Herencia). `DataLoader` invoca `read()` de forma uniforme
sobre cualquiera de estas subclases (pilar de Polimorfismo).
"""

from abc import ABC, abstractmethod

import pandas as pd

from exceptions import DataParsingError


class BaseFileReader(ABC):
    """Contrato abstracto para leer un archivo de datos como DataFrame."""

    @abstractmethod
    def read(self, path: str) -> pd.DataFrame:
        """Lee el archivo en `path` y devuelve su contenido como DataFrame.

        Debe lanzar `DataParsingError` si el contenido no puede
        interpretarse en el formato esperado por la subclase concreta.
        """
        raise NotImplementedError


class CsvReader(BaseFileReader):
    """Lee archivos CSV usando pandas."""

    def read(self, path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            raise DataParsingError(f"No se pudo interpretar el archivo CSV '{path}': {exc}") from exc


class JsonReader(BaseFileReader):
    """Lee archivos JSON orientados a registros (lista de objetos) usando pandas."""

    def read(self, path: str) -> pd.DataFrame:
        try:
            return pd.read_json(path, orient="records")
        except ValueError as exc:
            raise DataParsingError(f"No se pudo interpretar el archivo JSON '{path}': {exc}") from exc


class XlsxReader(BaseFileReader):
    """Lee archivos XLSX (hoja principal) usando pandas con el motor openpyxl."""

    def read(self, path: str) -> pd.DataFrame:
        try:
            return pd.read_excel(path, engine="openpyxl")
        except ValueError as exc:
            raise DataParsingError(f"No se pudo interpretar el archivo XLSX '{path}': {exc}") from exc
