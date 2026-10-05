"""Separación de los clientes en conjuntos de entrenamiento y prueba.

Ver `specs/008-preprocesamiento-modelo/`.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from exceptions import ModelPreparationError


class TrainTestSplitter:
    """Separa los datos en entrenamiento y prueba conservando la proporción de clases.

    La separación es estratificada por la variable objetivo (la proporción de
    aprobados queda igual en ambos conjuntos) y reproducible gracias a una
    semilla fija.

    Args:
        test_size: Proporción de filas que van al conjunto de prueba.
        random_state: Semilla para que la separación sea siempre la misma.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Separa `X` e `y` en entrenamiento y prueba.

        Args:
            X: Variables de entrada, una fila por cliente.
            y: Variable objetivo, alineada fila a fila con `X`.

        Returns:
            `(X_train, X_test, y_train, y_test)`, cada uno con índice
            `0..n-1`. La correspondencia fila a fila entre `X` e `y` se
            conserva. No modifica `X` ni `y`.

        Raises:
            ModelPreparationError: Si alguna clase de `y` tiene menos de 2
                elementos, porque entonces no se puede conservar la proporción.
        """
        for clase, cantidad in y.value_counts().items():
            if cantidad < 2:
                raise ModelPreparationError(
                    "No se puede separar conservando la proporción: "
                    f"la clase {clase} tiene {cantidad} cliente(s) (se necesitan al menos 2)."
                )

        partes = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        X_train, X_test, y_train, y_test = (parte.reset_index(drop=True) for parte in partes)
        return X_train, X_test, y_train, y_test
