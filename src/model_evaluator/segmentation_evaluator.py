"""Evaluación de la segmentación K-means: silhouette y perfiles de segmentos.

Ver `specs/010-model-evaluator/`.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from model_trainer import ClientSegmentation, TrainingResult


class SegmentationEvaluator:
    """Evalúa si los segmentos son razonables y describe cada uno.

    - **Silhouette** (de -1 a 1): qué tan separados están los segmentos. Se
      calcula para varios números de segmentos, para justificar cuántos usar.
    - **Perfiles**: por segmento, número de clientes, promedio de cada
      variable en su escala original (edad en años, ingreso en pesos) y tasa
      de aprobación. Es lo que convierte los segmentos en acciones de negocio.

    Args:
        random_state: Semilla de los K-means que se entrenan para comparar.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def silhouette(
        self,
        X: pd.DataFrame,
        segmentacion: ClientSegmentation,
        valores_k: tuple[int, ...] = (2, 3, 4, 5),
    ) -> pd.DataFrame:
        """Calcula el silhouette de K-means para cada número de segmentos.

        Args:
            X: Variables de entrada (normalmente `X_train`).
            segmentacion: La segmentación entrenada, para marcar su `k`.
            valores_k: Números de segmentos a probar. Se omiten los que no
                son menores que el número de clientes.

        Returns:
            Un DataFrame con columnas `k`, `silhouette` y `es_el_modelo`.
        """
        filas = []
        for k in valores_k:
            if k >= len(X):
                continue
            etiquetas = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit_predict(X)
            filas.append(
                {
                    "k": k,
                    "silhouette": float(silhouette_score(X, etiquetas)),
                    "es_el_modelo": k == segmentacion.n_segmentos,
                }
            )
        return pd.DataFrame(filas, columns=["k", "silhouette", "es_el_modelo"])

    @staticmethod
    def silhouette_modelo(X: pd.DataFrame, segmentacion: ClientSegmentation) -> float:
        """Silhouette de los segmentos del modelo entrenado sobre `X`."""
        return float(silhouette_score(X, segmentacion.predict(X)))

    @staticmethod
    def k_sugerido(tabla: pd.DataFrame) -> int:
        """Número de segmentos con mayor silhouette."""
        return int(tabla.loc[tabla["silhouette"].idxmax(), "k"])

    @staticmethod
    def perfiles(resultado: TrainingResult) -> pd.DataFrame:
        """Describe cada segmento en valores entendibles, con los clientes de entrenamiento y prueba.

        Deshace el escalado de la feature 008 (`x * desviación + media`) para
        mostrar cada variable en su escala original. Las variables binarias
        quedan como proporción (p. ej. 0.40 = 40% con vivienda propia).

        Args:
            resultado: Resultado de `ModelTrainer.train`.

        Returns:
            Un DataFrame con índice `segmento` y columnas `n_clientes`, una
            por variable (promedio en escala original) y `tasa_aprobacion`.
        """
        datos = resultado.datos
        X = pd.concat([datos.X_train, datos.X_test], ignore_index=True)
        for columna, (media, desviacion) in datos.parametros_escalado.items():
            X[columna] = X[columna] * desviacion + media

        ids = datos.ids_train + datos.ids_test
        X["segmento"] = resultado.segmentos.loc[ids].to_numpy()
        X["credito_aprobado"] = pd.concat([datos.y_train, datos.y_test], ignore_index=True).to_numpy()

        agrupado = X.groupby("segmento")
        perfiles = agrupado[datos.variables].mean()
        perfiles.insert(0, "n_clientes", agrupado.size())
        perfiles["tasa_aprobacion"] = agrupado["credito_aprobado"].mean()
        return perfiles
