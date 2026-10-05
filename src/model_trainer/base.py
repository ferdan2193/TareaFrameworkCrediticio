"""Clase base de los modelos supervisados y validaciones comunes.

Ver `specs/009-model-trainer/`.
"""

import time

import pandas as pd

from exceptions import InvalidTrainingDataError, ModelNotTrainedError


def validar_X(X: pd.DataFrame) -> None:
    """Verifica que `X` sirva para entrenar o predecir.

    Args:
        X: Variables de entrada, normalmente las de `ModelData`.

    Raises:
        InvalidTrainingDataError: Si `X` está vacío, tiene columnas que no
            son numéricas o tiene valores faltantes.
    """
    if X.empty:
        raise InvalidTrainingDataError("Datos de entrenamiento inválidos: la tabla está vacía")
    for columna in X.columns:
        if not pd.api.types.is_numeric_dtype(X[columna]):
            raise InvalidTrainingDataError(
                f"Datos de entrenamiento inválidos: la columna '{columna}' no es numérica"
            )
        if X[columna].isna().any():
            raise InvalidTrainingDataError(
                f"Datos de entrenamiento inválidos: la columna '{columna}' tiene valores faltantes"
            )


def validar_columnas(X: pd.DataFrame, variables: list[str]) -> None:
    """Verifica que `X` tenga las mismas columnas, en el mismo orden, que en entrenamiento.

    Raises:
        InvalidTrainingDataError: Si las columnas no coinciden.
    """
    if list(X.columns) != variables:
        raise InvalidTrainingDataError("Las columnas no coinciden con las de entrenamiento")


class SupervisedModel:
    """Base común de los modelos que predicen `credito_aprobado`.

    Reúne lo que comparten todos los modelos supervisados: validar los datos,
    medir el tiempo de entrenamiento y entregar predicciones como `pd.Series`.
    Cada subclase solo define su `nombre`, sus `hiperparametros` y cómo crear
    su estimador (`_crear_estimador`).

    Args:
        random_state: Semilla, para que el entrenamiento sea reproducible.

    Attributes:
        variables: Columnas con las que se entrenó (vacía antes de `fit`).
        tiempo_entrenamiento: Segundos que tardó `fit` (`None` antes de `fit`).
        estimador: Estimador de `scikit-learn` ya entrenado (`None` antes de
            `fit`). `ModelEvaluator` puede usarlo para leer coeficientes o
            importancias de variables.
    """

    nombre = "Modelo supervisado"

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.variables: list[str] = []
        self.tiempo_entrenamiento: float | None = None
        self.estimador = None

    @property
    def hiperparametros(self) -> dict[str, object]:
        """Hiperparámetros con los que se crea el estimador."""
        raise NotImplementedError

    @property
    def entrenado(self) -> bool:
        """`True` si el modelo ya se entrenó con `fit`."""
        return self.estimador is not None

    def _crear_estimador(self):
        """Crea el estimador de `scikit-learn` sin entrenar."""
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SupervisedModel":
        """Entrena el modelo.

        Args:
            X: Variables de entrada de entrenamiento (numéricas, sin faltantes).
            y: `credito_aprobado` de entrenamiento (0/1), alineado con `X`.

        Returns:
            La misma instancia, ya entrenada.

        Raises:
            InvalidTrainingDataError: Si `X` no es válido, `X` e `y` tienen
                distinto largo, o `y` tiene una sola clase.
        """
        validar_X(X)
        if len(X) != len(y):
            raise InvalidTrainingDataError(
                f"X tiene {len(X)} filas pero y tiene {len(y)}; deben coincidir"
            )
        if y.nunique() < 2:
            raise InvalidTrainingDataError(
                "Se necesitan al menos 2 clases en y para entrenar; "
                f"solo hay: {sorted(y.unique().tolist())}"
            )

        estimador = self._crear_estimador()
        inicio = time.perf_counter()
        estimador.fit(X, y)
        self.tiempo_entrenamiento = time.perf_counter() - inicio
        self.estimador = estimador
        self.variables = list(X.columns)
        return self

    def _validar_prediccion(self, X: pd.DataFrame) -> None:
        if not self.entrenado:
            raise ModelNotTrainedError(f"{self.nombre}: llama a fit antes de predecir")
        validar_X(X)
        validar_columnas(X, self.variables)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predice si se aprueba el crédito (1) o no (0).

        Args:
            X: Variables de entrada con las mismas columnas que en `fit`.

        Returns:
            Una `pd.Series` de enteros 0/1 llamada `"prediccion"`, con el
            índice de `X`.

        Raises:
            ModelNotTrainedError: Si el modelo no está entrenado.
            InvalidTrainingDataError: Si `X` no es válido o sus columnas no
                coinciden con las de entrenamiento.
        """
        self._validar_prediccion(X)
        return pd.Series(self.estimador.predict(X).astype(int), index=X.index, name="prediccion")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Calcula la probabilidad de que se apruebe el crédito.

        Args:
            X: Variables de entrada con las mismas columnas que en `fit`.

        Returns:
            Una `pd.Series` de valores entre 0 y 1 llamada
            `"probabilidad_aprobacion"`, con el índice de `X`.

        Raises:
            ModelNotTrainedError: Si el modelo no está entrenado.
            InvalidTrainingDataError: Si `X` no es válido o sus columnas no
                coinciden con las de entrenamiento.
        """
        self._validar_prediccion(X)
        columna_aprobado = list(self.estimador.classes_).index(1)
        return pd.Series(
            self.estimador.predict_proba(X)[:, columna_aprobado],
            index=X.index,
            name="probabilidad_aprobacion",
        )
