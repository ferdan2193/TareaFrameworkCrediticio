"""Modelo de regresión logística para predecir `credito_aprobado`.

Ver `specs/009-model-trainer/`.
"""

from sklearn.linear_model import LogisticRegression

from .base import SupervisedModel


class LogisticRegressionModel(SupervisedModel):
    """Regresión logística: el modelo base e interpretable del proyecto.

    Es el estándar de la industria en scoring crediticio: es lineal, se
    explica fácilmente al negocio y sus coeficientes indican cuánto pesa cada
    variable en la probabilidad de aprobación.

    Args:
        random_state: Semilla, para que el entrenamiento sea reproducible.
    """

    nombre = "Regresión logística"

    @property
    def hiperparametros(self) -> dict[str, object]:
        """`C=1.0` (regularización por defecto), `max_iter=1000` y la semilla."""
        return {"C": 1.0, "max_iter": 1000, "random_state": self.random_state}

    def _crear_estimador(self) -> LogisticRegression:
        return LogisticRegression(**self.hiperparametros)
