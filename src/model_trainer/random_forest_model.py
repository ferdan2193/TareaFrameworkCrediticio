"""Modelo random forest para predecir `credito_aprobado`.

Ver `specs/009-model-trainer/`.
"""

from sklearn.ensemble import RandomForestClassifier

from .base import SupervisedModel


class RandomForestModel(SupervisedModel):
    """Random forest: modelo no lineal basado en árboles de decisión.

    Combina muchos árboles, así que puede aprender reglas con varios umbrales
    a la vez, como la regla real de aprobación ("edad entre 18 y 69 **y**
    score > 600 **y** ..."). Además, indica la importancia de cada variable.

    Args:
        random_state: Semilla, para que el entrenamiento sea reproducible.
        n_estimators: Número de árboles (100 por defecto).
    """

    nombre = "Random forest"

    def __init__(self, random_state: int = 42, n_estimators: int = 100) -> None:
        super().__init__(random_state)
        self.n_estimators = n_estimators

    @property
    def hiperparametros(self) -> dict[str, object]:
        """Número de árboles y semilla."""
        return {"n_estimators": self.n_estimators, "random_state": self.random_state}

    def _crear_estimador(self) -> RandomForestClassifier:
        return RandomForestClassifier(**self.hiperparametros)
