"""Métricas de desempeño de los modelos de clasificación.

Ver `specs/010-model-evaluator/`.
"""

from dataclasses import asdict, dataclass, field

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from exceptions import EvaluationError

DESCRIPCION_METRICAS: dict[str, str] = {
    "exactitud": (
        "Proporción de clientes clasificados correctamente (aprobados y rechazados). Es fácil "
        "de comunicar, pero trata igual los dos tipos de error, aunque en crédito no cuestan "
        "lo mismo."
    ),
    "precision": (
        "De los clientes que el modelo aprueba, qué proporción realmente cumple la política de "
        "crédito. Es la métrica principal del proyecto: un falso positivo (aprobar a quien no "
        "cumple) es el error más costoso, porque es dinero prestado con riesgo."
    ),
    "recall": (
        "De los clientes que cumplen la política, qué proporción aprueba el modelo. Un recall "
        "bajo significa rechazar a buenos clientes: negocio perdido."
    ),
    "f1": (
        "Media armónica de precisión y recall. Resume en un solo número el equilibrio entre no "
        "aprobar de más y no rechazar de más; se usa como desempate."
    ),
    "auc_roc": (
        "Probabilidad de que el modelo asigne mayor probabilidad de aprobación a un cliente que "
        "cumple que a uno que no cumple, sin depender del umbral de 0.5. 0.5 equivale al azar "
        "y 1.0 a una separación perfecta."
    ),
    "matriz_confusion": (
        "Cuenta los cuatro resultados posibles: verdaderos positivos (aprobados bien), falsos "
        "positivos (aprobados que no cumplían), verdaderos negativos (rechazados bien) y falsos "
        "negativos (buenos clientes rechazados). Muestra qué tipo de error comete el modelo."
    ),
    "silhouette": (
        "Mide qué tan separados están los segmentos de K-means, de -1 a 1: cerca de 1, cada "
        "cliente está claramente en su grupo; cerca de 0, los grupos se traslapan. Sirve para "
        "elegir el número de segmentos."
    ),
}
"""Qué mide cada métrica y por qué importa en una decisión de crédito."""


@dataclass(frozen=True)
class MetricasModelo:
    """Métricas de un modelo de clasificación, con "aprobado" (1) como clase positiva.

    Attributes:
        exactitud, precision, recall, f1: Métricas entre 0 y 1.
        auc_roc: Área bajo la curva ROC, o `None` si no se pudo calcular.
        verdaderos_negativos, falsos_positivos, falsos_negativos,
            verdaderos_positivos: Matriz de confusión.
        notas: Avisos sobre casos especiales (p. ej. sin aprobaciones predichas).
    """

    exactitud: float
    precision: float
    recall: float
    f1: float
    auc_roc: float | None
    verdaderos_negativos: int
    falsos_positivos: int
    falsos_negativos: int
    verdaderos_positivos: int
    notas: list[str] = field(default_factory=list)

    def como_dict(self) -> dict[str, float | int | None]:
        """Devuelve las métricas como diccionario, sin las notas."""
        valores = asdict(self)
        valores.pop("notas")
        return valores


class ClassificationMetrics:
    """Calcula las métricas de un modelo comparando sus predicciones con la realidad."""

    @staticmethod
    def calcular(
        y_real: pd.Series, y_pred: pd.Series, y_prob: pd.Series | None = None
    ) -> MetricasModelo:
        """Calcula exactitud, precisión, recall, F1, AUC-ROC y la matriz de confusión.

        Args:
            y_real: `credito_aprobado` real (0/1).
            y_pred: Predicción del modelo (0/1), alineada con `y_real`.
            y_prob: Probabilidad de aprobación predicha (opcional; necesaria
                para el AUC-ROC).

        Returns:
            Un `MetricasModelo`. Si el modelo no predijo ninguna aprobación, la
            precisión es 0 y se agrega una nota. Si no hay probabilidades o
            `y_real` tiene una sola clase, `auc_roc` es `None`.

        Raises:
            EvaluationError: Si `y_real` e `y_pred` tienen distinto largo.
        """
        if len(y_real) != len(y_pred):
            raise EvaluationError("y_real y y_pred deben tener el mismo número de elementos")

        notas = []
        if (pd.Series(y_pred) == 1).sum() == 0:
            notas.append("El modelo no predijo ninguna aprobación; la precisión se reporta como 0.")

        auc = None
        if y_prob is not None:
            if pd.Series(y_real).nunique() < 2:
                notas.append("AUC-ROC no disponible: el conjunto evaluado tiene una sola clase.")
            else:
                auc = float(roc_auc_score(y_real, y_prob))

        vn, fp, fn, vp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
        return MetricasModelo(
            exactitud=float(accuracy_score(y_real, y_pred)),
            precision=float(precision_score(y_real, y_pred, pos_label=1, zero_division=0)),
            recall=float(recall_score(y_real, y_pred, pos_label=1, zero_division=0)),
            f1=float(f1_score(y_real, y_pred, pos_label=1, zero_division=0)),
            auc_roc=auc,
            verdaderos_negativos=int(vn),
            falsos_positivos=int(fp),
            falsos_negativos=int(fn),
            verdaderos_positivos=int(vp),
            notas=notas,
        )
