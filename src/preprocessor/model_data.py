"""Resultado de la preparación de datos para el modelo.

`ModelData` es lo que devuelve `Preprocessor.prepare_for_model` (ver
`specs/008-preprocesamiento-modelo/data-model.md`) y lo que recibe
`ModelTrainer`.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ModelData:
    """Conjuntos de entrenamiento y prueba listos para el modelo, con su trazabilidad.

    Attributes:
        X_train: Variables de entrada de entrenamiento: sin faltantes,
            numéricas, imputadas, codificadas y escaladas.
        X_test: Variables de entrada de prueba, con las mismas columnas y el
            mismo orden que `X_train`, transformadas con los parámetros de
            entrenamiento.
        y_train: `credito_aprobado` (0/1) de entrenamiento.
        y_test: `credito_aprobado` (0/1) de prueba.
        ids_train: `ID_Cliente` de cada fila de `X_train`, en el mismo orden.
        ids_test: `ID_Cliente` de cada fila de `X_test`, en el mismo orden.
        variables: Columnas finales de `X_train` y `X_test`.
        descartadas: Columnas que no llegaron al modelo, con el motivo.
        valores_imputacion: Valor con el que se rellenó cada columna.
        parametros_escalado: `(media, desviación)` de cada columna escalada.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    ids_train: list[str]
    ids_test: list[str]
    variables: list[str]
    descartadas: dict[str, str]
    valores_imputacion: dict[str, object]
    parametros_escalado: dict[str, tuple[float, float]]

    def as_tuple(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Devuelve `(X_train, X_test, y_train, y_test)`, el formato del diagrama."""
        return self.X_train, self.X_test, self.y_train, self.y_test
