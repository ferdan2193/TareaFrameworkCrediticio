"""Punto de entrada público del componente Preprocessor.

`Preprocessor` es la única clase que un desarrollador que consuma el
framework necesita conocer para combinar las fuentes de datos ya limpias
por `DataValidator` en una sola tabla por cliente (ver
`specs/005-preprocesador-fusion-clientes/contracts/`), y para calcular la
variable objetivo `credito_aprobado` con la regla de negocio del proyecto (ver
`specs/007-preprocessor-credito-aprobado/`).
"""

import pandas as pd
from matplotlib.figure import Figure

from data_visualizer import DataVisualizer
from exceptions import MissingRuleColumnsError, ModelPreparationError

from .encoder import CategoricalEncoder
from .imputer import DataImputer
from .model_data import ModelData
from .scaler import FeatureScaler
from .splitter import TrainTestSplitter


class Preprocessor:
    """Combina las fuentes de datos del proyecto y define la variable objetivo."""

    EDAD_MINIMA_APROBACION = 18
    """Edad mínima para aprobar crédito (inclusiva)."""

    EDAD_MAXIMA_APROBACION = 69
    """Edad máxima para aprobar crédito (inclusiva)."""

    SCORE_MINIMO_APROBACION = 600
    """Score de buró que hay que superar para aprobar crédito (exclusivo: `> 600`)."""

    COLUMNAS_REGLA = [
        "ID_Cliente",
        "Edad",
        "Score_Buro",
        "Mantiene_Morosidad_Previa",
        "Frecuencia_Deposito",
    ]
    """Columnas que necesita `define_credit_approval`."""

    COLUMNAS_HISTORIAL_DESCARTADAS = {"Moneda": "moneda única (MXN)"}
    """Columnas del historial que no pasan a la tabla por cliente, con el motivo."""

    def merge_sources(
        self,
        profile_df: pd.DataFrame,
        history_df: pd.DataFrame,
        bureau_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combina perfil, historial y buró en una sola tabla, por cliente.

        Args:
            profile_df: DataFrame del perfil del solicitante, ya limpio por
                `DataValidator`, con la columna `ID_Cliente`.
            history_df: DataFrame del historial transaccional, ya limpio
                por `DataValidator`, con la columna `ID_Cliente`.
            bureau_df: DataFrame del buró de crédito, ya limpio por
                `DataValidator`, con la columna `ID_Cliente`.

        Returns:
            Un nuevo DataFrame (no modifica ninguna de las tres entradas)
            con una fila por cada `ID_Cliente` presente en cualquiera de las
            tres fuentes (unión), todas las columnas originales sin alterar,
            y una columna nueva `credito_aprobado` (`int`) con valor `0` en
            todas las filas. Si a un cliente le faltan datos de alguna
            fuente, esas columnas quedan como `NaN` (se muestran como "ND"):
            no se rellenan con `0`, para no inventar datos. Imputar los
            faltantes es un paso posterior y explícito del pipeline.
        """
        combined = profile_df.merge(history_df, on="ID_Cliente", how="outer").merge(
            bureau_df, on="ID_Cliente", how="outer"
        )
        combined["credito_aprobado"] = 0
        return combined

    def show_combined_table(self, df: pd.DataFrame, title: str | None = None) -> Figure:
        """Muestra `df` como una tabla, reutilizando `DataVisualizer.mostrar_tabla`.

        Args:
            df: Normalmente el resultado de `merge_sources`, aunque acepta
                cualquier DataFrame.
            title: Título opcional mostrado sobre la tabla.

        Returns:
            Una `Figure` de matplotlib con una tabla que muestra todas las
            filas y columnas de `df`. Si `df` está vacío, la tabla solo
            muestra los encabezados, sin lanzar ningún error.
        """
        return DataVisualizer().mostrar_tabla(df, title)

    @staticmethod
    def _validar_columnas(df: pd.DataFrame, columnas: list[str]) -> None:
        """Verifica que `df` tenga todas las `columnas` indicadas.

        Raises:
            MissingRuleColumnsError: Si falta alguna; el mensaje las nombra en
                el orden de `columnas`.
        """
        faltantes = [c for c in columnas if c not in df.columns]
        if faltantes:
            raise MissingRuleColumnsError(
                "Faltan columnas necesarias para calcular credito_aprobado: "
                + ", ".join(faltantes)
            )

    def define_credit_approval(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula `credito_aprobado` con la regla de negocio del proyecto.

        Un cliente queda aprobado (`1`) solo si cumple las cuatro condiciones:

        1. `EDAD_MINIMA_APROBACION <= Edad <= EDAD_MAXIMA_APROBACION` (18 a 69).
        2. `Score_Buro > SCORE_MINIMO_APROBACION` (mayor que 600).
        3. `Mantiene_Morosidad_Previa == 0` (sin morosidad previa).
        4. Alguna de sus filas tiene `Frecuencia_Deposito > 0` (ha recibido
           transacciones).

        En cualquier otro caso queda en `0`. Un dato faltante o no numérico
        nunca cumple una condición, así que un cliente con información
        incompleta no se aprueba. Todas las filas de un mismo `ID_Cliente`
        reciben el mismo valor: si alguna fila suya no cumple, no se aprueba.

        Args:
            df: Tabla combinada, normalmente el resultado de `merge_sources`,
                con las columnas de `COLUMNAS_REGLA`.

        Returns:
            Un nuevo DataFrame (no modifica `df`) con las mismas filas, en el
            mismo orden, y la columna `credito_aprobado` (`int`, `0` o `1`)
            creada o reemplazada.

        Raises:
            MissingRuleColumnsError: Si falta alguna columna de `COLUMNAS_REGLA`.
        """
        self._validar_columnas(df, self.COLUMNAS_REGLA)
        df = df.copy()
        if df.empty:
            df["credito_aprobado"] = pd.Series(dtype=int)
            return df

        edad = pd.to_numeric(df["Edad"], errors="coerce")
        score = pd.to_numeric(df["Score_Buro"], errors="coerce")
        morosidad = pd.to_numeric(df["Mantiene_Morosidad_Previa"], errors="coerce")
        frecuencia = pd.to_numeric(df["Frecuencia_Deposito"], errors="coerce")
        por_cliente = df["ID_Cliente"]

        edad_ok = edad.between(self.EDAD_MINIMA_APROBACION, self.EDAD_MAXIMA_APROBACION)
        score_ok = score > self.SCORE_MINIMO_APROBACION
        sin_morosidad = morosidad == 0
        con_depositos = (frecuencia > 0).groupby(por_cliente).transform("any")

        cumple = edad_ok & score_ok & sin_morosidad & con_depositos
        df["credito_aprobado"] = cumple.groupby(por_cliente).transform("all").astype(int)
        return df

    def approval_summary(self, df: pd.DataFrame) -> pd.Series:
        """Cuenta cuántos clientes quedaron aprobados y cuántos no.

        Sirve para saber, antes de entrenar un modelo, si la variable
        objetivo tiene ambas clases y qué tan desbalanceada está.

        Args:
            df: Tabla con `ID_Cliente` y `credito_aprobado`, normalmente el
                resultado de `define_credit_approval`.

        Returns:
            Una `pd.Series` llamada `"clientes"`, con índice `[0, 1]` (siempre
            ambos, aunque alguno tenga conteo cero) y el número de clientes
            distintos en cada clase. Un cliente con varias filas cuenta una
            sola vez.

        Raises:
            MissingRuleColumnsError: Si falta `ID_Cliente` o `credito_aprobado`.
        """
        self._validar_columnas(df, ["ID_Cliente", "credito_aprobado"])
        por_cliente = df.drop_duplicates(subset="ID_Cliente")
        return (
            por_cliente["credito_aprobado"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
            .rename("clientes")
        )

    def aggregate_by_client(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deja una sola fila por cliente, resumiendo su historial transaccional.

        La tabla combinada tiene una fila por transacción, así que un cliente
        con varias transacciones aparece varias veces. Antes de separar en
        entrenamiento y prueba hay que dejar una fila por cliente; si no, un
        mismo cliente podría quedar en ambos conjuntos.

        Columnas del historial que se resumen:

        - `Monto_Total`: suma de `Monto_Transaccion` (`NaN` si todos faltan).
        - `Saldo_Promedio_Mensual`: promedio.
        - `Frecuencia_Deposito_Max`: máximo de `Frecuencia_Deposito`.
        - `Num_Transacciones`: filas con `Frecuencia_Deposito` presente
          (`0` si el cliente no tiene historial).

        `Moneda` se descarta (ver `COLUMNAS_HISTORIAL_DESCARTADAS`). Las demás
        columnas (perfil, buró, `credito_aprobado`) toman su primer valor,
        porque ya son iguales en todas las filas del cliente.

        Args:
            df: Tabla combinada, normalmente el resultado de
                `define_credit_approval`.

        Returns:
            Un nuevo DataFrame (no modifica `df`) con una fila por
            `ID_Cliente`, ordenado por `ID_Cliente` y con índice `0..n-1`.

        Raises:
            ModelPreparationError: Si falta `ID_Cliente`.
        """
        if "ID_Cliente" not in df.columns:
            raise ModelPreparationError(
                "No se puede resumir por cliente: falta la columna ID_Cliente"
            )

        por_cliente = df.groupby("ID_Cliente", sort=True)
        resumen = {}
        for columna in df.columns:
            if columna == "ID_Cliente" or columna in self.COLUMNAS_HISTORIAL_DESCARTADAS:
                continue
            if columna == "Monto_Transaccion":
                resumen["Monto_Total"] = por_cliente[columna].sum(min_count=1)
            elif columna == "Saldo_Promedio_Mensual":
                resumen[columna] = por_cliente[columna].mean()
            elif columna == "Frecuencia_Deposito":
                resumen["Frecuencia_Deposito_Max"] = por_cliente[columna].max()
                resumen["Num_Transacciones"] = por_cliente[columna].count()
            else:
                resumen[columna] = por_cliente[columna].first()

        return pd.DataFrame(resumen).reset_index()

    def prepare_for_model(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> ModelData:
        """Prepara la tabla etiquetada para entrenar y evaluar un modelo.

        Ejecuta, en este orden:

        1. `aggregate_by_client`: una fila por cliente.
        2. Separa `X` (todo menos `ID_Cliente` y `credito_aprobado`) e `y`
           (`credito_aprobado`).
        3. `TrainTestSplitter`: entrenamiento y prueba, estratificado por `y`.
        4. Quita las columnas constantes en entrenamiento.
        5. `DataImputer`, `CategoricalEncoder` y `FeatureScaler`, cada uno
           ajustado (`fit`) **solo** con entrenamiento y aplicado a ambos
           conjuntos, para no filtrar información del conjunto de prueba.

        No entrena ningún modelo: eso le corresponde a `ModelTrainer`.

        Args:
            df: Tabla etiquetada, normalmente el resultado de
                `define_credit_approval`.
            test_size: Proporción de clientes que van a prueba.
            random_state: Semilla de la separación, para que sea reproducible.

        Returns:
            Un `ModelData` con `X_train`, `X_test`, `y_train`, `y_test`, los
            `ID_Cliente` de cada conjunto y los parámetros aprendidos. No
            modifica `df`.

        Raises:
            ModelPreparationError: Si `df` está vacío, le falta `ID_Cliente` o
                `credito_aprobado`, o alguna clase tiene menos de 2 clientes.
        """
        if df.empty:
            raise ModelPreparationError("No se puede preparar para el modelo: la tabla está vacía")
        faltantes = [c for c in ("ID_Cliente", "credito_aprobado") if c not in df.columns]
        if faltantes:
            raise ModelPreparationError(
                "No se puede preparar para el modelo: faltan las columnas " + ", ".join(faltantes)
            )

        descartadas = {
            columna: motivo
            for columna, motivo in self.COLUMNAS_HISTORIAL_DESCARTADAS.items()
            if columna in df.columns
        }
        por_cliente = self.aggregate_by_client(df)
        X = por_cliente.drop(columns=["credito_aprobado"])  # ID_Cliente se quita tras separar
        y = por_cliente["credito_aprobado"].astype(int)

        X_train, X_test, y_train, y_test = TrainTestSplitter(test_size, random_state).split(X, y)
        ids_train = X_train.pop("ID_Cliente").tolist()
        ids_test = X_test.pop("ID_Cliente").tolist()

        constantes = [c for c in X_train.columns if X_train[c].nunique(dropna=True) <= 1]
        for columna in constantes:
            descartadas[columna] = "constante"
        X_train = X_train.drop(columns=constantes)
        X_test = X_test.drop(columns=constantes)

        imputador = DataImputer().fit(X_train)
        for columna in imputador.columnas_descartadas:
            descartadas[columna] = "sin valores en entrenamiento"
        X_train, X_test = imputador.transform(X_train), imputador.transform(X_test)

        codificador = CategoricalEncoder().fit(X_train)
        X_train, X_test = codificador.transform(X_train), codificador.transform(X_test)

        escalador = FeatureScaler().fit(X_train)
        X_train, X_test = escalador.transform(X_train), escalador.transform(X_test)

        return ModelData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            ids_train=ids_train,
            ids_test=ids_test,
            variables=list(X_train.columns),
            descartadas=descartadas,
            valores_imputacion=imputador.valores_imputacion,
            parametros_escalado=escalador.parametros,
        )
