"""Punto de entrada público del componente DataVisualizer.

`DataVisualizer` es la única clase que un desarrollador que consuma el
framework necesita conocer para generar visualizaciones exploratorias de
los datos ya limpiados por `DataValidator` (ver
`specs/003-datavisualizer-graficas/contracts/`). Cada método devuelve un
`matplotlib.figure.Figure` en vez de mostrarlo directamente, para no
depender de un backend interactivo ni de un entorno de notebook
específico.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.table import Table


class DataVisualizer:
    """Agrupa las visualizaciones exploratorias de las fuentes de datos del proyecto."""

    def mostrar_tabla(self, df: pd.DataFrame, titulo: str | None = None) -> Figure:
        """Genera una figura con una tabla del `DataFrame` recibido.

        Args:
            df: Cualquiera de los tres DataFrame limpios (buró de crédito,
                historial transaccional, perfil del solicitante).
            titulo: Título opcional mostrado sobre la tabla.

        Returns:
            Una `Figure` de matplotlib con una tabla que muestra todas las
            filas y columnas de `df`. Los valores faltantes (`NaN`/`None`)
            se muestran como `"ND"`. Si `df` está vacío, la tabla solo
            muestra los encabezados, sin lanzar ningún error.
        """
        fig, ax = plt.subplots()
        ax.axis("off")
        if titulo:
            ax.set_title(titulo)

        columnas = list(df.columns)
        if df.empty:
            # ax.table() no admite cellText=[] (falla con IndexError), así que la
            # tabla de solo-encabezados se construye con la API de bajo nivel.
            tabla = Table(ax, loc="center")
            ancho = 1.0 / len(columnas) if columnas else 1.0
            for indice_columna, nombre_columna in enumerate(columnas):
                tabla.add_cell(0, indice_columna, width=ancho, height=0.1, text=nombre_columna, loc="center")
            ax.add_table(tabla)
        else:
            ax.table(
                cellText=df.astype(object).where(df.notna(), "ND").astype(str).values.tolist(),
                colLabels=columnas,
                loc="center",
            )
        return fig

    def graficar_creditos_vs_score(self, df_buro: pd.DataFrame) -> Figure:
        """Genera una figura de dispersión de créditos activos vs. score de buró.

        Args:
            df_buro: DataFrame limpio del buró de crédito con las columnas
                `Creditos_Activos` y `Score_Buro`.

        Returns:
            Una `Figure` de matplotlib con un punto por fila de `df_buro`.
            Si `df_buro` está vacío, la figura no tiene puntos.
        """
        fig, ax = plt.subplots()
        ax.scatter(df_buro["Creditos_Activos"], df_buro["Score_Buro"])
        ax.set_xlabel("Créditos activos")
        ax.set_ylabel("Score de buró")
        return fig

    def graficar_score_vs_ingreso(
        self, df_buro: pd.DataFrame, df_perfil: pd.DataFrame
    ) -> Figure:
        """Genera una figura de dispersión de score de buró vs. ingreso declarado.

        Cruza `df_buro` y `df_perfil` por `ID_Cliente` (coincidencia exacta
        de texto); los clientes sin coincidencia exacta en ambos DataFrame
        quedan excluidos de la figura.

        Args:
            df_buro: DataFrame limpio del buró de crédito con `ID_Cliente`
                y `Score_Buro`.
            df_perfil: DataFrame limpio del perfil del solicitante con
                `ID_Cliente` e `Ingreso_Declarado`.

        Returns:
            Una `Figure` de matplotlib con un punto por cada `ID_Cliente`
            coincidente. Si no hay ninguna coincidencia, la figura no
            tiene puntos.
        """
        clientes_cruzados = pd.merge(
            df_buro[["ID_Cliente", "Score_Buro"]],
            df_perfil[["ID_Cliente", "Ingreso_Declarado"]],
            on="ID_Cliente",
            how="inner",
        )

        fig, ax = plt.subplots()
        ax.scatter(clientes_cruzados["Score_Buro"], clientes_cruzados["Ingreso_Declarado"])
        ax.set_xlabel("Score de buró")
        ax.set_ylabel("Ingreso declarado")
        return fig

    def graficar_distribucion_tipo_vivienda(self, df_perfil: pd.DataFrame) -> Figure:
        """Genera una figura de barras con la distribución de tipo de vivienda.

        Args:
            df_perfil: DataFrame limpio del perfil del solicitante con
                `Tipo_Vivienda` codificada como `0`/`1` por `DataValidator`.

        Returns:
            Una `Figure` de matplotlib con barras etiquetadas
            `"Vivienda propia"` y `"Otro tipo de vivienda"` (no `0`/`1`).
        """
        etiquetas = df_perfil["Tipo_Vivienda"].map(
            {1: "Vivienda propia", 0: "Otro tipo de vivienda"}
        )
        conteo = etiquetas.value_counts()

        fig, ax = plt.subplots()
        ax.bar(conteo.index, conteo.values)
        ax.set_xlabel("Tipo de vivienda")
        ax.set_ylabel("Cantidad de solicitantes")
        return fig
