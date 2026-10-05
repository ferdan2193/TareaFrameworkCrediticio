"""Escalado de variables numéricas a media 0 y desviación 1.

Ver `specs/008-preprocesamiento-modelo/`.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from exceptions import ModelPreparationError


class FeatureScaler:
    """Lleva las variables numéricas a una escala comparable.

    Cada variable numérica no binaria se transforma con
    `(x - media) / desviación`, usando la media y la desviación del conjunto
    de entrenamiento (aprendidas en `fit`). Las variables binarias (solo 0 y
    1) no se escalan, para que sigan siendo interpretables.

    Attributes:
        columnas_escaladas: Columnas que se escalan.
        parametros: `(media, desviación)` de cada columna escalada.
    """

    def __init__(self) -> None:
        self.columnas_escaladas: list[str] = []
        self.parametros: dict[str, tuple[float, float]] = {}
        self._ajustado = False
        self._escalador: StandardScaler | None = None

    @staticmethod
    def _es_binaria(serie: pd.Series) -> bool:
        return set(serie.dropna().unique()) <= {0, 1}

    def fit(self, X: pd.DataFrame) -> "FeatureScaler":
        """Aprende la media y la desviación de las columnas numéricas no binarias.

        Args:
            X: Conjunto de entrenamiento, sin faltantes.

        Returns:
            La misma instancia.
        """
        self.columnas_escaladas = [
            c
            for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c]) and not self._es_binaria(X[c])
        ]
        self.parametros = {}
        self._escalador = None
        if self.columnas_escaladas:
            self._escalador = StandardScaler().set_output(transform="pandas")
            self._escalador.fit(X[self.columnas_escaladas])
            self.parametros = {
                columna: (float(media), float(desviacion))
                for columna, media, desviacion in zip(
                    self.columnas_escaladas, self._escalador.mean_, self._escalador.scale_
                )
            }
        self._ajustado = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Escala las columnas aprendidas en `fit` con los parámetros de entrenamiento.

        Args:
            X: Cualquier conjunto con las mismas columnas que el usado en `fit`.

        Returns:
            Un nuevo DataFrame con el mismo orden de columnas y el índice de `X`.

        Raises:
            ModelPreparationError: Si se llama antes de `fit`.
        """
        if not self._ajustado:
            raise ModelPreparationError("FeatureScaler: llama a fit antes de transform")

        resultado = X.copy()
        if self._escalador is not None:
            escaladas = self._escalador.transform(X[self.columnas_escaladas])
            escaladas.index = X.index
            resultado[self.columnas_escaladas] = escaladas
        return resultado

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Equivale a `fit(X).transform(X)`."""
        return self.fit(X).transform(X)
