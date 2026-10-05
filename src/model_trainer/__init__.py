"""Componente ModelTrainer: entrena los modelos supervisados y la segmentación de clientes (ver specs/009-model-trainer/)."""

from .base import SupervisedModel
from .logistic_model import LogisticRegressionModel
from .random_forest_model import RandomForestModel
from .segmentation import ClientSegmentation
from .trainer import ModelTrainer
from .training_result import TrainingResult

__all__ = [
    "ModelTrainer",
    "TrainingResult",
    "SupervisedModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "ClientSegmentation",
]
