"""Segmentación de clientes con K-means (modelo no supervisado).

Ver `specs/009-model-trainer/`.
"""

import time

import pandas as pd
from sklearn.cluster import KMeans

from exceptions import InvalidTrainingDataError, ModelNotTrainedError

from .base import validar_columnas, validar_X


class ClientSegmentation:
    """Agrupa a los clientes en segmentos con perfiles parecidos, sin usar la etiqueta.

    Usa K-means: el algoritmo de segmentación más conocido, que funciona bien
    con datos escalados y produce grupos fáciles de explicar al negocio (cada
    segmento tiene un "cliente típico", su centro). Por defecto forma 3
    segmentos, pensando en perfiles de riesgo bajo, medio y alto.

    **No** recibe `credito_aprobado`: `fit` solo acepta las variables de
    entrada, así que los segmentos pueden revelar grupos que la regla de
    negocio no distingue.

    Args:
        n_segmentos: Número de segmentos (3 por defecto).
        random_state: Semilla, para que la segmentación sea reproducible.

    Attributes:
        variables: Columnas usadas (vacía antes de `fit`).
        tiempo_entrenamiento: Segundos que tardó `fit`.
        centros: Centro de cada segmento (una fila por segmento, una columna
            por variable), en la escala de los datos preparados.
        estimador: Estimador `KMeans` ya entrenado.
    """

    nombre = "Segmentación K-means"

    def __init__(self, n_segmentos: int = 3, random_state: int = 42) -> None:
        self.n_segmentos = n_segmentos
        self.random_state = random_state
        self.variables: list[str] = []
        self.tiempo_entrenamiento: float | None = None
        self.centros: pd.DataFrame | None = None
        self.estimador: KMeans | None = None

    @property
    def hiperparametros(self) -> dict[str, object]:
        """Número de segmentos, repeticiones de la inicialización (10) y semilla."""
        return {"n_clusters": self.n_segmentos, "n_init": 10, "random_state": self.random_state}

    def fit(self, X: pd.DataFrame) -> "ClientSegmentation":
        """Aprende los segmentos a partir de las variables de entrada.

        Args:
            X: Variables de entrada de entrenamiento (numéricas, sin faltantes).

        Returns:
            La misma instancia, ya entrenada.

        Raises:
            InvalidTrainingDataError: Si `X` no es válido o hay más segmentos
                que clientes.
        """
        validar_X(X)
        if self.n_segmentos > len(X):
            raise InvalidTrainingDataError(
                f"No se pueden formar {self.n_segmentos} segmentos con {len(X)} clientes"
            )

        estimador = KMeans(**self.hiperparametros)
        inicio = time.perf_counter()
        estimador.fit(X)
        self.tiempo_entrenamiento = time.perf_counter() - inicio
        self.estimador = estimador
        self.variables = list(X.columns)
        self.centros = pd.DataFrame(estimador.cluster_centers_, columns=self.variables)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Asigna a cada cliente el segmento aprendido más cercano.

        Args:
            X: Variables de entrada con las mismas columnas que en `fit`.

        Returns:
            Una `pd.Series` de enteros `0..n_segmentos-1` llamada `"segmento"`,
            con el índice de `X`.

        Raises:
            ModelNotTrainedError: Si la segmentación no está entrenada.
            InvalidTrainingDataError: Si `X` no es válido o sus columnas no
                coinciden con las de entrenamiento.
        """
        if self.estimador is None:
            raise ModelNotTrainedError(f"{self.nombre}: llama a fit antes de predecir")
        validar_X(X)
        validar_columnas(X, self.variables)
        return pd.Series(self.estimador.predict(X).astype(int), index=X.index, name="segmento")
