"""Resultado de la evaluación de los modelos.

`EvaluationResult` es lo que devuelve `ModelEvaluator.evaluate` (ver
`specs/010-model-evaluator/data-model.md`) y lo que usa `DataVisualizer` para
las gráficas del reporte.
"""

from dataclasses import dataclass

import pandas as pd

from .comparison import Recomendacion
from .metrics import MetricasModelo


@dataclass(frozen=True)
class EvaluationResult:
    """Todo lo que produjo la evaluación, listo para el reporte.

    Attributes:
        metricas: Métricas de prueba de cada modelo supervisado.
        validacion_cruzada: Tabla media/desviación de cada modelo.
        comparacion: Tabla comparativa (una fila por modelo).
        recomendacion: Modelo recomendado y su justificación.
        importancias: Importancia de variables de cada modelo.
        silhouette: Silhouette de K-means para cada número de segmentos.
        silhouette_modelo: Silhouette de la segmentación entrenada.
        k_sugerido: Número de segmentos con mayor silhouette.
        perfiles_segmentos: Perfil de cada segmento en escala original.
        descripcion_metricas: Qué mide cada métrica y por qué importa en
            crédito (texto base para justificar las métricas en el reporte).
    """

    metricas: dict[str, MetricasModelo]
    validacion_cruzada: dict[str, pd.DataFrame]
    comparacion: pd.DataFrame
    recomendacion: Recomendacion
    importancias: dict[str, pd.DataFrame]
    silhouette: pd.DataFrame
    silhouette_modelo: float
    k_sugerido: int
    perfiles_segmentos: pd.DataFrame
    descripcion_metricas: dict[str, str]
