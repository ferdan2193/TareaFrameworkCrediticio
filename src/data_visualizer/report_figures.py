"""Generación y guardado de todas las gráficas analíticas para el reporte.

Ver `specs/011-datavisualizer-seaborn/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from data_validator import ReporteCalidad
    from model_evaluator import EvaluationResult
    from model_trainer import TrainingResult

from .exploratory_charts import ExploratoryCharts
from .grafica import GraficaInterpretada
from .model_charts import ModelCharts
from .quality_charts import QualityCharts
from .segmentation_charts import SegmentationCharts


class ReportFigures:
    """Genera las 11 gráficas del reporte y las guarda como imágenes.

    Orden: calidad de cada fuente, distribución de clases, correlaciones
    (mapa y barras contra la aprobación), matriz de confusión, curva ROC,
    importancia de variables, silhouette y PCA de segmentos. Carpeta sugerida para guardarlas: `reporte/figuras/`.
    """

    @staticmethod
    def generar(
        calidad: dict[str, tuple[ReporteCalidad, ReporteCalidad]],
        tabla_por_cliente: pd.DataFrame,
        resultado: TrainingResult,
        evaluacion: EvaluationResult,
    ) -> list[GraficaInterpretada]:
        """Genera todas las gráficas con su interpretación.

        Args:
            calidad: Revisión de calidad antes y después, por fuente.
            tabla_por_cliente: Tabla etiquetada con una fila por cliente
                (`Preprocessor.aggregate_by_client`).
            resultado: Resultado de `ModelTrainer.train`.
            evaluacion: Resultado de `ModelEvaluator.evaluate`.

        Returns:
            La lista de gráficas, en el orden del reporte.
        """
        graficas = [QualityCharts.calidad(antes, despues) for antes, despues in calidad.values()]
        graficas += [
            ExploratoryCharts.distribucion_clases(tabla_por_cliente),
            ExploratoryCharts.correlaciones(tabla_por_cliente),
            ExploratoryCharts.correlacion_con_aprobacion(tabla_por_cliente),
            ModelCharts.matriz_confusion(evaluacion),
            ModelCharts.curva_roc(resultado),
            ModelCharts.importancia(evaluacion),
            SegmentationCharts.silhouette(evaluacion),
            SegmentationCharts.pca_segmentos(resultado),
        ]
        return graficas

    @staticmethod
    def guardar_todas(
        graficas: list[GraficaInterpretada], carpeta: str | Path
    ) -> list[GraficaInterpretada]:
        """Guarda cada gráfica como PNG en `carpeta` (que se crea si no existe).

        Returns:
            Las gráficas con `ruta` llenada.
        """
        return [grafica.guardar(carpeta) for grafica in graficas]
