"""Punto de entrada público del componente Preprocessor.

`Preprocessor` es la única clase que un desarrollador que consuma el
framework necesita conocer para combinar las fuentes de datos ya limpias
por `DataValidator` en una sola tabla por cliente (ver
`specs/005-preprocesador-fusion-clientes/contracts/`).
"""

import pandas as pd
from matplotlib.figure import Figure

from data_visualizer import DataVisualizer


class Preprocessor:
    """Combina las fuentes de datos del proyecto en una sola tabla por cliente."""

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
            tres fuentes (unión), todas las columnas originales sin alterar
            salvo por el relleno de huecos donde a un cliente le faltaban
            datos de alguna fuente (`0` en columnas numéricas, cadena vacía
            en columnas de texto), y una columna nueva `credito_aprobado`
            (`int`) con valor `0` en todas las filas.
        """
        combined = profile_df.merge(history_df, on="ID_Cliente", how="outer").merge(
            bureau_df, on="ID_Cliente", how="outer"
        )
        for column in combined.columns:
            if column == "ID_Cliente":
                continue
            if pd.api.types.is_numeric_dtype(combined[column]):
                combined[column] = combined[column].fillna(0)
            else:
                combined[column] = combined[column].fillna("")
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
