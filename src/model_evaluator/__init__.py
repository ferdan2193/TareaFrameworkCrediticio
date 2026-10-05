"""Componente ModelEvaluator: métricas, comparación de modelos y evaluación de la segmentación (ver specs/010-model-evaluator/)."""

from .comparison import ModelComparator, Recomendacion
from .cross_validation import CrossValidator
from .evaluation_result import EvaluationResult
from .evaluator import ModelEvaluator
from .importance import VARIABLES_DE_LA_REGLA, FeatureImportance
from .metrics import DESCRIPCION_METRICAS, ClassificationMetrics, MetricasModelo
from .segmentation_evaluator import SegmentationEvaluator

__all__ = [
    "ModelEvaluator",
    "EvaluationResult",
    "ClassificationMetrics",
    "MetricasModelo",
    "DESCRIPCION_METRICAS",
    "CrossValidator",
    "ModelComparator",
    "Recomendacion",
    "FeatureImportance",
    "VARIABLES_DE_LA_REGLA",
    "SegmentationEvaluator",
]
