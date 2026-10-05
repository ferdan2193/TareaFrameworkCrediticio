"""Importancia de variables de los modelos supervisados.

Ver `specs/010-model-evaluator/`.
"""

import pandas as pd

from exceptions import EvaluationError
from model_trainer import SupervisedModel

VARIABLES_DE_LA_REGLA = [
    "Edad",
    "Score_Buro",
    "Mantiene_Morosidad_Previa",
    "Frecuencia_Deposito_Max",
]
"""Variables con las que se calcula `credito_aprobado` (feature 007)."""


class FeatureImportance:
    """Indica qué variables pesan más en cada modelo.

    - **Regresión logística**: el coeficiente de cada variable. Como las
      variables están estandarizadas, los coeficientes son comparables: un
      valor positivo aumenta la probabilidad de aprobación y uno negativo la
      reduce. La `magnitud` es su valor absoluto.
    - **Random forest**: la importancia de cada variable (cuánto ayuda a
      separar a aprobados de no aprobados). Siempre es positiva y suman 1.

    La columna `de_la_regla` marca las variables de la regla de aprobación,
    para comprobar si el modelo las "descubrió".
    """

    @staticmethod
    def calcular(modelo: SupervisedModel) -> pd.DataFrame:
        """Calcula la importancia de variables de un modelo entrenado.

        Args:
            modelo: Modelo supervisado entrenado (`LogisticRegressionModel` o
                `RandomForestModel`).

        Returns:
            Un DataFrame con columnas `variable`, `valor`, `magnitud` y
            `de_la_regla`, ordenado de mayor a menor `magnitud`.

        Raises:
            EvaluationError: Si el estimador no tiene coeficientes ni
                importancias de variables.
        """
        estimador = modelo.estimador
        if hasattr(estimador, "coef_"):
            valores = list(estimador.coef_[0])
        elif hasattr(estimador, "feature_importances_"):
            valores = list(estimador.feature_importances_)
        else:
            raise EvaluationError(
                f"No se reconoce cómo obtener la importancia de variables del modelo '{modelo.nombre}'"
            )

        tabla = pd.DataFrame(
            {
                "variable": modelo.variables,
                "valor": [float(v) for v in valores],
                "magnitud": [abs(float(v)) for v in valores],
                "de_la_regla": [v in VARIABLES_DE_LA_REGLA for v in modelo.variables],
            }
        )
        return tabla.sort_values("magnitud", ascending=False, kind="stable").reset_index(drop=True)
