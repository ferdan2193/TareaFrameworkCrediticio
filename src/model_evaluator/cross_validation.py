"""Validación cruzada de los modelos supervisados.

Ver `specs/010-model-evaluator/`.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from exceptions import EvaluationError
from model_trainer import SupervisedModel


class CrossValidator:
    """Estima el desempeño de un modelo repitiendo el entrenamiento en varias particiones.

    Con solo 7 clientes de prueba, un error cambia la exactitud en 14 puntos.
    La validación cruzada divide el conjunto de entrenamiento en
    `n_particiones` partes que conservan la proporción de aprobados: entrena
    con todas menos una, evalúa en la que queda y repite. La media y la
    desviación de esas evaluaciones son una estimación más estable.

    Entrena **copias nuevas** del estimador en cada partición, así que el
    modelo original no cambia.

    Args:
        n_particiones: Número de particiones (5 por defecto).
        random_state: Semilla para mezclar y repartir los clientes.
    """

    def __init__(self, n_particiones: int = 5, random_state: int = 42) -> None:
        self.n_particiones = n_particiones
        self.random_state = random_state

    def evaluar(self, modelo: SupervisedModel, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Ejecuta la validación cruzada de un modelo ya entrenado (usa una copia).

        Args:
            modelo: Modelo supervisado del `TrainingResult`.
            X: Variables de entrada (normalmente `X_train`).
            y: Variable objetivo alineada con `X`.

        Returns:
            Un DataFrame con índice `exactitud`, `precision`, `recall`, `f1` y
            columnas `media` y `desviacion`.

        Raises:
            EvaluationError: Si alguna clase tiene menos clientes que particiones.
        """
        for clase, cantidad in y.value_counts().items():
            if cantidad < self.n_particiones:
                raise EvaluationError(
                    f"No se puede hacer validación cruzada con {self.n_particiones} particiones: "
                    f"la clase {clase} tiene {cantidad} clientes"
                )

        particiones = StratifiedKFold(
            n_splits=self.n_particiones, shuffle=True, random_state=self.random_state
        )
        metricas = {
            "exactitud": "accuracy",
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
        }
        resultados = cross_validate(clone(modelo.estimador), X, y, cv=particiones, scoring=metricas)

        return pd.DataFrame(
            {
                "media": [float(np.mean(resultados[f"test_{m}"])) for m in metricas],
                "desviacion": [float(np.std(resultados[f"test_{m}"])) for m in metricas],
            },
            index=list(metricas),
        )
