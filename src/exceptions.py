"""Jerarquía de excepciones compartida por todo el framework.

Módulo único y centralizado para las excepciones de los distintos componentes
del pipeline (`DataLoader`, `DataValidator`, y los que se agreguen más
adelante), de forma que agregar, renombrar o reorganizar excepciones —
incluyendo el nombre de la clase base— se haga en un solo lugar.
"""

import logging
import sys

_logger = logging.getLogger("framework")
_logger.setLevel(logging.ERROR)

if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[FRAMEWORK ERROR] %(message)s"))
    _logger.addHandler(_handler)


class FrameworkError(Exception):
    """Clase base de todas las excepciones propias del framework.

    Permite capturar cualquier error del framework con un único
    `except FrameworkError`, sin necesidad de conocer el componente ni el
    tipo específico que lo originó.

    Al construirse, registra automáticamente un mensaje claro en consola
    (tipo de excepción + descripción) para toda la jerarquía, incluidas las
    subclases que se agreguen en el futuro.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        _logger.error("%s: %s", type(self).__name__, self)


class DataLoaderError(FrameworkError):
    """Clase base de los errores del componente DataLoader.

    Permite capturar cualquier error de carga con `except DataLoaderError`,
    sin necesidad de conocer el tipo específico.
    """


class MissingFilePathError(DataLoaderError):
    """Se lanza cuando no se proporciona una ruta de archivo (`path` vacío o None)."""


class FileNotFoundInSourceError(DataLoaderError):
    """Se lanza cuando la ruta indicada no corresponde a un archivo existente."""


class UnsupportedFileFormatError(DataLoaderError):
    """Se lanza cuando la extensión del archivo no está soportada (csv, json, xlsx)."""


class DataParsingError(DataLoaderError):
    """Se lanza cuando el archivo tiene un formato soportado pero su contenido
    no puede interpretarse (corrupto o mal formado)."""
