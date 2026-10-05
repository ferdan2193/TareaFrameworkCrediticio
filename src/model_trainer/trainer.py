"""Punto de entrada público del componente ModelTrainer.

Ver `specs/009-model-trainer/`.
"""

import pandas as pd

from preprocessor import ModelData

from .logistic_model import LogisticRegressionModel
from .random_forest_model import RandomForestModel
from .segmentation import ClientSegmentation
from .training_result import TrainingResult


class ModelTrainer:
    """Entrena los modelos del proyecto a partir de los datos ya preparados.

    Entrena dos modelos supervisados (regresión logística y random forest)
    para predecir `credito_aprobado`, y una segmentación K-means de los
    clientes. **No** calcula métricas ni compara modelos: eso le corresponde a
    `ModelEvaluator`.

    Args:
        random_state: Semilla para los tres modelos (42 por defecto).
        n_segmentos: Número de segmentos de K-means (3 por defecto).
    """

    def __init__(self, random_state: int = 42, n_segmentos: int = 3) -> None:
        self.random_state = random_state
        self.n_segmentos = n_segmentos

    def train(self, datos: ModelData) -> TrainingResult:
        """Entrena los dos modelos supervisados y la segmentación.

        1. `LogisticRegressionModel` y `RandomForestModel` con
           `datos.X_train` y `datos.y_train`.
        2. `ClientSegmentation` solo con `datos.X_train` (sin la etiqueta).
        3. Asigna un segmento a cada cliente de entrenamiento y de prueba.

        Args:
            datos: Resultado de `Preprocessor.prepare_for_model`.

        Returns:
            Un `TrainingResult` con los modelos, la segmentación y el segmento
            de cada cliente. No modifica `datos`.
        """
        modelos = [
            LogisticRegressionModel(self.random_state),
            RandomForestModel(self.random_state),
        ]
        for modelo in modelos:
            modelo.fit(datos.X_train, datos.y_train)

        segmentacion = ClientSegmentation(self.n_segmentos, self.random_state).fit(datos.X_train)
        segmentos = pd.concat(
            [segmentacion.predict(datos.X_train), segmentacion.predict(datos.X_test)],
            ignore_index=True,
        )
        segmentos.index = pd.Index(datos.ids_train + datos.ids_test, name="ID_Cliente")

        return TrainingResult(
            modelos={modelo.nombre: modelo for modelo in modelos},
            segmentacion=segmentacion,
            segmentos=segmentos,
            variables=list(datos.variables),
            datos=datos,
        )
