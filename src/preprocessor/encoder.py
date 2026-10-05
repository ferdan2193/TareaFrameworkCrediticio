"""Codificación de variables categóricas en columnas numéricas 0/1.

Ver `specs/008-preprocesamiento-modelo/`.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from exceptions import ModelPreparationError


class CategoricalEncoder:
    """Convierte cada columna de texto en una columna 0/1 por categoría.

    Las categorías se aprenden en `fit` (solo con el entrenamiento). Una
    categoría que aparece después y no se vio en entrenamiento queda con `0`
    en todas las columnas de esa variable, sin error. Si no hay columnas de
    texto, `transform` devuelve los datos sin cambios.

    Attributes:
        columnas_categoricas: Columnas de texto que se codifican.
        categorias: Categorías vistas en entrenamiento, por columna.
    """

    def __init__(self) -> None:
        self.columnas_categoricas: list[str] = []
        self.categorias: dict[str, list] = {}
        self._ajustado = False
        self._codificador: OneHotEncoder | None = None

    def fit(self, X: pd.DataFrame) -> "CategoricalEncoder":
        """Aprende las categorías de las columnas de texto de `X`.

        Args:
            X: Conjunto de entrenamiento.

        Returns:
            La misma instancia.
        """
        self.columnas_categoricas = [
            c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
        ]
        self.categorias = {}
        self._codificador = None
        if self.columnas_categoricas:
            self._codificador = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ).set_output(transform="pandas")
            self._codificador.fit(X[self.columnas_categoricas])
            self.categorias = {
                columna: list(categorias)
                for columna, categorias in zip(
                    self.columnas_categoricas, self._codificador.categories_
                )
            }
        self._ajustado = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reemplaza las columnas de texto por columnas `<columna>_<categoría>`.

        Args:
            X: Cualquier conjunto con las mismas columnas que el usado en `fit`.

        Returns:
            Un nuevo DataFrame numérico con el índice de `X`: primero las
            columnas que no eran de texto y después las codificadas.

        Raises:
            ModelPreparationError: Si se llama antes de `fit`.
        """
        if not self._ajustado:
            raise ModelPreparationError("CategoricalEncoder: llama a fit antes de transform")
        if self._codificador is None:
            return X.copy()

        codificadas = self._codificador.transform(X[self.columnas_categoricas]).astype(int)
        codificadas.index = X.index
        resto = X.drop(columns=self.columnas_categoricas)
        return pd.concat([resto, codificadas], axis=1)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Equivale a `fit(X).transform(X)`."""
        return self.fit(X).transform(X)
