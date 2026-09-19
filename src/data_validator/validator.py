"""Punto de entrada público del componente DataValidator.

`DataValidator` es la única clase que un desarrollador que consuma el
framework necesita conocer para limpiar y normalizar los datos ya cargados
por `DataLoader`. No
expone ninguna interfaz visual ni de consola: es una API puramente
programática que recibe y devuelve `pandas.DataFrame`.
"""

import numpy as np
import pandas as pd


class DataValidator:
    """Agrupa las reglas de limpieza de las tres fuentes de datos del proyecto.

    Args:
        tasa_cambio_usd_mxn: Tasa usada para convertir montos en USD a su
            equivalente en MXN dentro de `limpiar_historial_transaccional`.
            Configurable para no depender de una constante fija dentro de
            la lógica de conversión.
    """

    def __init__(self, tasa_cambio_usd_mxn: float = 15) -> None:
        self.tasa_cambio_usd_mxn = tasa_cambio_usd_mxn

    def limpiar_buro_credito(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza `Mantiene_Morosidad_Previa` a únicamente `0` o `1`.

        Args:
            df: DataFrame del buró de crédito con la columna
                `Mantiene_Morosidad_Previa`.

        Returns:
            Un nuevo DataFrame (no modifica `df`) con
            `Mantiene_Morosidad_Previa` normalizada a `int` (`0` o `1`).
        """
        df = df.copy()
        df["Mantiene_Morosidad_Previa"] = (
            df["Mantiene_Morosidad_Previa"].map(self._normalizar_morosidad).astype(int)
        )
        df["ID_Cliente"] = df["ID_Cliente"].astype(str).str.replace(
            r"(?i)^cliente_", "CLI-", regex=True
        )

        return df

    @staticmethod
    def _normalizar_morosidad(valor: object) -> int:
        """Normaliza un único valor de `Mantiene_Morosidad_Previa` a `0`/`1`."""
        if isinstance(valor, str):
            texto = valor.strip().lower()
            return 1 if texto in ("si", "sí") else 0
        if pd.isna(valor):
            return 0
        return int(valor)

    def limpiar_historial_transaccional(self, df: pd.DataFrame) -> pd.DataFrame:
        """Completa nulos y normaliza moneda/montos del historial transaccional.

        Args:
            df: DataFrame del historial transaccional con las columnas
                `Monto_Transaccion`, `Saldo_Promedio_Mensual` y `Moneda`.

        Returns:
            Un nuevo DataFrame (no modifica `df`) sin nulos en esos tres
            campos, con los montos originalmente en USD convertidos a MXN
            usando `self.tasa_cambio_usd_mxn`, y `Moneda` actualizada a
            `"MXN"` en esos registros.
        """
        with pd.option_context("future.no_silent_downcasting", True):
            df = df.fillna(
                {
                    "Saldo_Promedio_Mensual": 0,
                    "Monto_Transaccion": 0,
                    "Moneda": "MXN",
                }
            ).infer_objects(copy=False)

        es_usd = df["Moneda"].astype(str).str.strip().str.upper() == "USD"
        df.loc[es_usd, "Monto_Transaccion"] *= self.tasa_cambio_usd_mxn
        df.loc[es_usd, "Saldo_Promedio_Mensual"] *= self.tasa_cambio_usd_mxn
        df.loc[es_usd, "Moneda"] = "MXN"

        return df

    def limpiar_perfil_solicitante(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza el perfil del solicitante sin eliminar registros.

        Args:
            df: DataFrame del perfil del solicitante con las columnas
                `Edad`, `Ingreso_Declarado`, `Antiguedad_Laboral` y
                `Tipo_Vivienda`.

        Returns:
            Un nuevo DataFrame (no modifica `df`, y conserva todas sus
            filas) con `Edad` en `0` cuando el valor original estaba fuera
            de `[18, 70]` o era nulo/no numérico, sin nulos en
            `Ingreso_Declarado` ni `Antiguedad_Laboral`, y con
            `Tipo_Vivienda` codificada como `int` (`1` = vivienda propia,
            `0` = cualquier otro caso).
        """
        df = df.copy()
        edad = pd.to_numeric(df["Edad"], errors="coerce")
        edad_valida = edad.between(18, 70)
        df["Edad"] = edad.where(edad_valida, 0)

        with pd.option_context("future.no_silent_downcasting", True):
            df["Ingreso_Declarado"] = df["Ingreso_Declarado"].fillna(0).infer_objects(copy=False)
            df["Antiguedad_Laboral"] = df["Antiguedad_Laboral"].fillna(0).infer_objects(copy=False)

        es_propia = df["Tipo_Vivienda"].astype(str).str.strip().str.lower() == "propia"
        df["Tipo_Vivienda"] = np.where(es_propia, 1, 0)

        return df
