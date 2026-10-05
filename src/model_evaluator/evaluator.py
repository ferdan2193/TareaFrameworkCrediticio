"""Punto de entrada público del componente ModelEvaluator.

Ver `specs/010-model-evaluator/`.
"""

from exceptions import EvaluationError
from model_trainer import TrainingResult

from .comparison import ModelComparator
from .cross_validation import CrossValidator
from .evaluation_result import EvaluationResult
from .importance import FeatureImportance
from .metrics import DESCRIPCION_METRICAS, ClassificationMetrics
from .segmentation_evaluator import SegmentationEvaluator


class ModelEvaluator:
    """Evalúa los modelos entrenados y la segmentación de clientes.

    Flujo de `evaluate`:

    1. Métricas de prueba de cada modelo supervisado (con `X_test`/`y_test`).
    2. Validación cruzada de cada modelo sobre `X_train`/`y_train` (con copias).
    3. Tabla comparativa y recomendación (precisión → F1 → simplicidad).
    4. Importancia de variables de cada modelo.
    5. Silhouette de K-means y perfiles de los segmentos.

    Args:
        n_particiones: Particiones de la validación cruzada (5 por defecto).
        random_state: Semilla de la validación cruzada y de K-means.
    """

    def __init__(self, n_particiones: int = 5, random_state: int = 42) -> None:
        self.n_particiones = n_particiones
        self.random_state = random_state

    def evaluate(self, resultado: TrainingResult) -> EvaluationResult:
        """Evalúa todo a partir del resultado del entrenamiento.

        Args:
            resultado: Resultado de `ModelTrainer.train`.

        Returns:
            Un `EvaluationResult`. No modifica `resultado` ni sus modelos.

        Raises:
            EvaluationError: Si `resultado` no tiene modelos supervisados, o si
                la validación cruzada no se puede hacer.
        """
        if not resultado.modelos:
            raise EvaluationError("No hay modelos supervisados para evaluar")

        datos = resultado.datos
        validador = CrossValidator(self.n_particiones, self.random_state)
        metricas, validacion_cruzada, importancias = {}, {}, {}
        for nombre, modelo in resultado.modelos.items():
            metricas[nombre] = ClassificationMetrics.calcular(
                datos.y_test, modelo.predict(datos.X_test), modelo.predict_proba(datos.X_test)
            )
            validacion_cruzada[nombre] = validador.evaluar(modelo, datos.X_train, datos.y_train)
            importancias[nombre] = FeatureImportance.calcular(modelo)

        comparacion = ModelComparator.comparar(metricas, validacion_cruzada)

        evaluador_segmentos = SegmentationEvaluator(self.random_state)
        silhouette = evaluador_segmentos.silhouette(datos.X_train, resultado.segmentacion)

        return EvaluationResult(
            metricas=metricas,
            validacion_cruzada=validacion_cruzada,
            comparacion=comparacion,
            recomendacion=ModelComparator.recomendar(comparacion),
            importancias=importancias,
            silhouette=silhouette,
            silhouette_modelo=evaluador_segmentos.silhouette_modelo(
                datos.X_train, resultado.segmentacion
            ),
            k_sugerido=evaluador_segmentos.k_sugerido(silhouette),
            perfiles_segmentos=evaluador_segmentos.perfiles(resultado),
            descripcion_metricas=DESCRIPCION_METRICAS,
        )
