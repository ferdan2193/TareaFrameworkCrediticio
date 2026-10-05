"""Resultado del entrenamiento de los modelos.

`TrainingResult` es lo que devuelve `ModelTrainer.train` (ver
`specs/009-model-trainer/data-model.md`) y lo que usará `ModelEvaluator`.
"""

from dataclasses import dataclass

import pandas as pd

from preprocessor import ModelData

from .base import SupervisedModel
from .segmentation import ClientSegmentation


@dataclass(frozen=True)
class TrainingResult:
    """Modelos entrenados y segmentación de clientes.

    Attributes:
        modelos: Modelos supervisados entrenados, por nombre
            (`"Regresión logística"`, `"Random forest"`).
        segmentacion: Segmentación K-means entrenada.
        segmentos: Segmento de cada cliente (entrenamiento y prueba), con
            índice `ID_Cliente`.
        variables: Variables de entrada que usaron todos los modelos.
        datos: Los datos de entrada, para que `ModelEvaluator` use `X_test` e
            `y_test` sin volver a prepararlos.
    """

    modelos: dict[str, SupervisedModel]
    segmentacion: ClientSegmentation
    segmentos: pd.Series
    variables: list[str]
    datos: ModelData

    def resumen(self) -> pd.DataFrame:
        """Resume qué se entrenó, con qué hiperparámetros y cuánto tardó.

        Returns:
            Un DataFrame con una fila por modelo (los supervisados y la
            segmentación) y las columnas `modelo`, `tipo`, `hiperparametros`
            y `tiempo_entrenamiento_s`.
        """
        filas = [
            {
                "modelo": nombre,
                "tipo": "supervisado",
                "hiperparametros": modelo.hiperparametros,
                "tiempo_entrenamiento_s": round(modelo.tiempo_entrenamiento, 4),
            }
            for nombre, modelo in self.modelos.items()
        ]
        filas.append(
            {
                "modelo": "K-means",
                "tipo": "no supervisado",
                "hiperparametros": self.segmentacion.hiperparametros,
                "tiempo_entrenamiento_s": round(self.segmentacion.tiempo_entrenamiento, 4),
            }
        )
        return pd.DataFrame(filas)
