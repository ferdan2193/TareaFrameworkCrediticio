"""Punto de entrada público del componente DataValidator.

`DataValidator` es la única clase que un desarrollador que consuma el
framework necesita conocer para limpiar y normalizar los datos ya cargados
por `DataLoader`. No
expone ninguna interfaz visual ni de consola: es una API puramente
programática que recibe y devuelve `pandas.DataFrame`.
"""

import numpy as np
import pandas as pd

from exceptions import DataValidatorError, MissingColumnsError, UnknownSourceError

from .reporte import ReporteCalidad


class DataValidator:
    """Agrupa las reglas de limpieza de las tres fuentes de datos del proyecto.

    Args:
        tasa_cambio_usd_mxn: Tasa usada para convertir montos en USD a su
            equivalente en MXN dentro de `limpiar_historial_transaccional`.
            Configurable para no depender de una constante fija dentro de
            la lógica de conversión.
    """

    COLUMNAS_ESPERADAS: dict[str, list[str]] = {
        "buro_credito": [
            "ID_Cliente",
            "Score_Buro",
            "Creditos_Activos",
            "Mantiene_Morosidad_Previa",
        ],
        "historial_transaccional": [
            "ID_Cliente",
            "Monto_Transaccion",
            "Moneda",
            "Saldo_Promedio_Mensual",
            "Frecuencia_Deposito",
        ],
        "perfil_solicitante": [
            "ID_Cliente",
            "Edad",
            "Ingreso_Declarado",
            "Antiguedad_Laboral",
            "Tipo_Vivienda",
        ],
    }
    """Columnas que debe tener cada fuente de datos, por nombre de fuente."""

    EDAD_MINIMA = 18
    """Edad mínima válida (inclusiva); por debajo se considera atípica."""

    EDAD_MAXIMA = 70
    """Edad máxima válida (inclusiva); por encima se considera atípica."""

    def __init__(self, tasa_cambio_usd_mxn: float = 15) -> None:
        self.tasa_cambio_usd_mxn = tasa_cambio_usd_mxn

    def _validar_columnas(self, df: pd.DataFrame, fuente: str) -> None:
        """Verifica que `df` tenga todas las columnas esperadas de `fuente`.

        Raises:
            MissingColumnsError: Si falta alguna columna esperada; el mensaje
                nombra la fuente y las columnas faltantes.
        """
        faltantes = [c for c in self.COLUMNAS_ESPERADAS[fuente] if c not in df.columns]
        if faltantes:
            raise MissingColumnsError(
                f"La fuente '{fuente}' no tiene las columnas esperadas: {', '.join(faltantes)}"
            )

    @staticmethod
    def _eliminar_duplicados(df: pd.DataFrame, por_cliente: bool) -> pd.DataFrame:
        """Elimina filas idénticas y, si `por_cliente`, `ID_Cliente` repetidos.

        Se aplica después de normalizar, para detectar también duplicados que
        solo diferían en formato. Con `por_cliente` se conserva la primera
        aparición de cada `ID_Cliente`. El índice resultante es `0..n-1`.
        """
        df = df.drop_duplicates()
        if por_cliente:
            df = df.drop_duplicates(subset="ID_Cliente", keep="first")
        return df.reset_index(drop=True)

    def limpiar_buro_credito(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza el buró de crédito y elimina sus duplicados.

        Args:
            df: DataFrame del buró de crédito con la columna
                `Mantiene_Morosidad_Previa`.

        Returns:
            Un nuevo DataFrame (no modifica `df`) con
            `Mantiene_Morosidad_Previa` normalizada a `int` (`0` o `1`), el
            prefijo `CLIENTE_` de `ID_Cliente` cambiado a `CLI-`, sin filas
            duplicadas y con un solo registro por `ID_Cliente` (se conserva
            el primero).

        Raises:
            MissingColumnsError: Si falta alguna columna esperada del buró.
        """
        self._validar_columnas(df, "buro_credito")
        df = df.copy()
        df["Mantiene_Morosidad_Previa"] = (
            df["Mantiene_Morosidad_Previa"].map(self._normalizar_morosidad).astype(int)
        )
        df["ID_Cliente"] = df["ID_Cliente"].astype(str).str.replace(
            r"(?i)^cliente_", "CLI-", regex=True
        )

        return self._eliminar_duplicados(df, por_cliente=True)

    @staticmethod
    def _normalizar_morosidad(valor: object) -> int:
        """Normaliza un único valor de `Mantiene_Morosidad_Previa` a `0`/`1`.

        Acepta `"si"`/`"sí"`, `1`/`True` y también `"1"`/`"true"` escritos
        como texto, que es como Excel puede entregarlos.
        """
        if isinstance(valor, str):
            texto = valor.strip().lower()
            return 1 if texto in ("si", "sí", "1", "true") else 0
        if pd.isna(valor):
            return 0
        return int(valor)

    def limpiar_historial_transaccional(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza moneda/montos del historial transaccional y elimina duplicados.

        Args:
            df: DataFrame del historial transaccional con las columnas
                `ID_Cliente`, `Monto_Transaccion`, `Moneda`,
                `Saldo_Promedio_Mensual` y `Frecuencia_Deposito`.

        Returns:
            Un nuevo DataFrame (no modifica `df`) con `Moneda` nula
            completada como `"MXN"`, los montos originalmente en USD
            convertidos a MXN usando `self.tasa_cambio_usd_mxn` (y su `Moneda`
            actualizada a `"MXN"`), y sin filas idénticas. Un cliente puede
            tener varias transacciones. `Monto_Transaccion` y
            `Saldo_Promedio_Mensual` faltantes quedan como `NaN` (se muestran
            como "ND"), no como `0`.

        Raises:
            MissingColumnsError: Si falta alguna columna esperada del historial.
        """
        self._validar_columnas(df, "historial_transaccional")
        df = df.copy()
        for columna in ("Monto_Transaccion", "Saldo_Promedio_Mensual"):
            df[columna] = pd.to_numeric(df[columna], errors="coerce")
        df["Moneda"] = df["Moneda"].fillna("MXN")

        es_usd = df["Moneda"].astype(str).str.strip().str.upper() == "USD"
        df.loc[es_usd, "Monto_Transaccion"] *= self.tasa_cambio_usd_mxn
        df.loc[es_usd, "Saldo_Promedio_Mensual"] *= self.tasa_cambio_usd_mxn
        df.loc[es_usd, "Moneda"] = "MXN"

        return self._eliminar_duplicados(df, por_cliente=False)

    def limpiar_perfil_solicitante(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza el perfil del solicitante y elimina sus duplicados.

        Args:
            df: DataFrame del perfil del solicitante con las columnas
                `ID_Cliente`, `Edad`, `Ingreso_Declarado`,
                `Antiguedad_Laboral` y `Tipo_Vivienda`.

        Returns:
            Un nuevo DataFrame (no modifica `df`) con:

            - `Edad` como `NaN` (se muestra como "ND") cuando el valor
              original es atípico (fuera de `[EDAD_MINIMA, EDAD_MAXIMA]`,
              rango inclusivo), nulo o no numérico. El registro se conserva.
            - `Ingreso_Declarado` y `Antiguedad_Laboral` faltantes o no
              numéricos como `NaN`, no como `0`.
            - `Tipo_Vivienda` codificada como `int` (`1` = vivienda propia,
              `0` = cualquier otro caso).
            - Sin filas duplicadas y con un solo registro por `ID_Cliente`
              (se conserva el primero).

        Raises:
            MissingColumnsError: Si falta alguna columna esperada del perfil.
        """
        self._validar_columnas(df, "perfil_solicitante")
        df = df.copy()
        edad = pd.to_numeric(df["Edad"], errors="coerce")
        edad_valida = edad.between(self.EDAD_MINIMA, self.EDAD_MAXIMA)
        df["Edad"] = edad.where(edad_valida, np.nan)
        for columna in ("Ingreso_Declarado", "Antiguedad_Laboral"):
            df[columna] = pd.to_numeric(df[columna], errors="coerce")

        es_propia = df["Tipo_Vivienda"].astype(str).str.strip().str.lower() == "propia"
        df["Tipo_Vivienda"] = np.where(es_propia, 1, 0)

        return self._eliminar_duplicados(df, por_cliente=True)

    def revisar_calidad(self, df: pd.DataFrame, fuente: str) -> ReporteCalidad:
        """Revisa la estructura y calidad de una fuente de datos, sin modificarla.

        Funciona igual con los datos originales y con los ya limpios, para
        comparar el antes y el después. No lanza error por columnas faltantes
        ni por datos sucios: los reporta.

        Args:
            df: Tabla de la fuente a revisar.
            fuente: `"buro_credito"`, `"historial_transaccional"` o
                `"perfil_solicitante"`.

        Returns:
            Un `ReporteCalidad` con filas, columnas faltantes/adicionales,
            tipos, faltantes por columna, duplicados, `ID_Cliente` repetidos
            (perfil y buró) y edades atípicas (perfil).

        Raises:
            UnknownSourceError: Si `fuente` no es un nombre reconocido.
        """
        if fuente not in self.COLUMNAS_ESPERADAS:
            raise UnknownSourceError(
                f"Fuente '{fuente}' no reconocida. "
                f"Valores válidos: {', '.join(self.COLUMNAS_ESPERADAS)}"
            )

        esperadas = self.COLUMNAS_ESPERADAS[fuente]

        ids_repetidos = None
        if fuente in ("buro_credito", "perfil_solicitante") and "ID_Cliente" in df.columns:
            ids_repetidos = int(df["ID_Cliente"].duplicated().sum())

        edades_atipicas = None
        if fuente == "perfil_solicitante" and "Edad" in df.columns:
            edad = pd.to_numeric(df["Edad"], errors="coerce")
            fuera_de_rango = edad.notna() & ~edad.between(self.EDAD_MINIMA, self.EDAD_MAXIMA)
            edades_atipicas = int(fuera_de_rango.sum())

        return ReporteCalidad(
            fuente=fuente,
            filas=len(df),
            columnas_faltantes=[c for c in esperadas if c not in df.columns],
            columnas_adicionales=[c for c in df.columns if c not in esperadas],
            tipos={c: str(df[c].dtype) for c in df.columns},
            faltantes={c: int(df[c].isna().sum()) for c in df.columns},
            duplicados=int(df.duplicated().sum()),
            ids_repetidos=ids_repetidos,
            edades_atipicas=edades_atipicas,
        )

    @staticmethod
    def comparar_calidad(antes: ReporteCalidad, despues: ReporteCalidad) -> pd.DataFrame:
        """Compara la calidad de una fuente antes y después de limpiarla.

        Args:
            antes: Revisión de calidad de los datos originales.
            despues: Revisión de calidad de los datos limpios, de la misma fuente.

        Returns:
            Un DataFrame con índice `indicador` y columnas `antes` y
            `después`. Indicadores: `Filas`, `Duplicados`, `IDs repetidos` y
            `Edades atípicas` (solo si aplican a la fuente), `Columnas
            faltantes` y una fila `Faltantes: <columna>` por cada columna con
            faltantes antes o después.

        Raises:
            DataValidatorError: Si los dos reportes son de fuentes distintas.
        """
        if antes.fuente != despues.fuente:
            raise DataValidatorError(
                f"No se pueden comparar reportes de fuentes distintas: "
                f"'{antes.fuente}' y '{despues.fuente}'"
            )

        filas = {
            "Filas": (antes.filas, despues.filas),
            "Duplicados": (antes.duplicados, despues.duplicados),
        }
        if antes.ids_repetidos is not None or despues.ids_repetidos is not None:
            filas["IDs repetidos"] = (antes.ids_repetidos, despues.ids_repetidos)
        if antes.edades_atipicas is not None or despues.edades_atipicas is not None:
            filas["Edades atípicas"] = (antes.edades_atipicas, despues.edades_atipicas)
        filas["Columnas faltantes"] = (len(antes.columnas_faltantes), len(despues.columnas_faltantes))

        columnas = list(antes.faltantes) + [c for c in despues.faltantes if c not in antes.faltantes]
        for columna in columnas:
            valores = (antes.faltantes.get(columna, 0), despues.faltantes.get(columna, 0))
            if any(valores):
                filas[f"Faltantes: {columna}"] = valores

        tabla = pd.DataFrame.from_dict(filas, orient="index", columns=["antes", "después"])
        tabla.index.name = "indicador"
        return tabla

