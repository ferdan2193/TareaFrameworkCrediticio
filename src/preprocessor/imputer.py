"""Imputación de los datos faltantes ("ND").

Ver `specs/008-preprocesamiento-modelo/`.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from exceptions import ModelPreparationError


class DataImputer:
    """Reemplaza los faltantes con valores aprendidos del conjunto de entrenamiento.

    Las columnas numéricas se rellenan con su mediana y las de texto con su
    valor más frecuente. Los valores se aprenden en `fit` (solo con el
    entrenamiento) y se aplican igual en `transform` a cualquier conjunto,
    para no filtrar información del conjunto de prueba.

    Attributes:
        valores_imputacion: Valor usado para rellenar cada columna.
        columnas_descartadas: Columnas sin ningún valor en entrenamiento; no
            se puede calcular su mediana, así que se quitan.
    """

    def __init__(self) -> None:
        self.valores_imputacion: dict[str, object] = {}
        self.columnas_descartadas: list[str] = []
        self._columnas: list[str] | None = None
        self._imputadores: list[tuple[list[str], SimpleImputer]] = []

    @staticmethod
    def _con_nan(X: pd.DataFrame) -> pd.DataFrame:
        """Representa todos los faltantes como `np.nan`.

        `SimpleImputer` solo reconoce `np.nan`, y en columnas de texto pandas
        puede guardar un faltante como `None`.
        """
        return X.astype(object).where(X.notna(), np.nan).infer_objects()

    def fit(self, X: pd.DataFrame) -> "DataImputer":
        """Aprende la mediana o la moda de cada columna de `X`.

        Args:
            X: Conjunto de entrenamiento.

        Returns:
            La misma instancia, para poder encadenar `fit(...).transform(...)`.
        """
        self.columnas_descartadas = [c for c in X.columns if X[c].isna().all()]
        self._columnas = [c for c in X.columns if c not in self.columnas_descartadas]
        numericas = [c for c in self._columnas if pd.api.types.is_numeric_dtype(X[c])]
        texto = [c for c in self._columnas if c not in numericas]

        self._imputadores = []
        self.valores_imputacion = {}
        for columnas, estrategia in ((numericas, "median"), (texto, "most_frequent")):
            if not columnas:
                continue
            imputador = SimpleImputer(strategy=estrategia).set_output(transform="pandas")
            imputador.fit(self._con_nan(X[columnas]))
            self._imputadores.append((columnas, imputador))
            for columna, valor in zip(columnas, imputador.statistics_):
                self.valores_imputacion[columna] = (
                    float(valor) if estrategia == "median" else valor
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rellena los faltantes de `X` con los valores aprendidos en `fit`.

        Args:
            X: Cualquier conjunto (entrenamiento o prueba) con las mismas
                columnas que el usado en `fit`.

        Returns:
            Un nuevo DataFrame sin faltantes, sin las columnas descartadas, con
            el orden de columnas y el índice de `X`.

        Raises:
            ModelPreparationError: Si se llama antes de `fit`.
        """
        if self._columnas is None:
            raise ModelPreparationError("DataImputer: llama a fit antes de transform")

        resultado = X[self._columnas].copy()
        for columnas, imputador in self._imputadores:
            imputado = imputador.transform(self._con_nan(X[columnas]))
            imputado.index = X.index
            resultado[columnas] = imputado
        return resultado

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Equivale a `fit(X).transform(X)`."""
        return self.fit(X).transform(X)
