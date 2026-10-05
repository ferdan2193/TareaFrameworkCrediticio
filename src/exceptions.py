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


class DataValidatorError(FrameworkError):
    """Clase base de los errores del componente DataValidator.

    Permite capturar cualquier error de validación con
    `except DataValidatorError`, sin necesidad de conocer el tipo específico.
    """


class MissingColumnsError(DataValidatorError):
    """Se lanza cuando a una fuente de datos le faltan columnas esperadas."""


class UnknownSourceError(DataValidatorError):
    """Se lanza cuando el nombre de fuente de datos no es reconocido."""


class PreprocessorError(FrameworkError):
    """Clase base de los errores del componente Preprocessor.

    Permite capturar cualquier error del Preprocessor con
    `except PreprocessorError`.
    """


class MissingRuleColumnsError(PreprocessorError):
    """Se lanza cuando faltan columnas necesarias para calcular o resumir `credito_aprobado`."""


class ModelPreparationError(PreprocessorError):
    """Se lanza cuando los datos no se pueden preparar para el modelo.

    Por ejemplo: entrada vacía, faltan columnas, una clase con menos de 2
    clientes, o un paso usado antes de ajustarlo (`fit`).
    """


class ModelTrainerError(FrameworkError):
    """Clase base de los errores del componente ModelTrainer."""


class InvalidTrainingDataError(ModelTrainerError):
    """Se lanza cuando los datos no sirven para entrenar o predecir.

    Por ejemplo: datos vacíos, con texto o faltantes, con una sola clase, con
    más segmentos que clientes, o con columnas distintas a las de
    entrenamiento.
    """


class ModelNotTrainedError(ModelTrainerError):
    """Se lanza cuando se pide una predicción antes de entrenar el modelo."""


class ModelEvaluatorError(FrameworkError):
    """Clase base de los errores del componente ModelEvaluator."""


class EvaluationError(ModelEvaluatorError):
    """Se lanza cuando no se puede evaluar.

    Por ejemplo: entradas de distinto largo, pocos clientes por clase para la
    validación cruzada, un modelo sin importancia de variables reconocible o
    un resultado sin modelos.
    """
