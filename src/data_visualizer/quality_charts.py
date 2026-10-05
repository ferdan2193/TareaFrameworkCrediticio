"""Gráfica de calidad de datos antes y después de la limpieza.

Ver `specs/011-datavisualizer-seaborn/`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from data_validator import ReporteCalidad

from .estilo import AZUL, NARANJA, TINTA, TINTA_SECUNDARIA, nueva_figura, titulo
from .grafica import GraficaInterpretada

PALETA_MOMENTO = {"Antes": AZUL, "Después": NARANJA}
ORDEN_MOMENTO = ["Antes", "Después"]


def _barras(ax, datos: pd.DataFrame, x: str, y: str, orden: list[str]) -> None:
    sns.barplot(data=datos, x=x, y=y, hue="momento", order=orden, hue_order=ORDEN_MOMENTO,
                palette=PALETA_MOMENTO, ax=ax, width=0.7, saturation=1)
    for contenedor in ax.containers:
        ax.bar_label(contenedor, fmt="%d", color=TINTA_SECUNDARIA, fontsize=9, padding=2)
    ax.grid(axis="x", visible=False)
    # Son conteos: el eje empieza en 0 y deja espacio para las etiquetas.
    ax.set_ylim(0, max(1, float(datos[y].max()) * 1.15))


class QualityCharts:
    """Muestra con evidencia visual qué resolvió la limpieza de datos."""

    @staticmethod
    def calidad(antes: ReporteCalidad, despues: ReporteCalidad) -> GraficaInterpretada:
        """Faltantes por columna, duplicados y edades atípicas, antes y después de limpiar.

        Args:
            antes: Revisión de calidad de los datos originales.
            despues: Revisión de calidad de los datos limpios.

        Returns:
            Una gráfica con dos paneles: faltantes por columna, y duplicados (más
            edades atípicas, si la fuente las tiene).
        """
        fuente = antes.fuente
        figura, (ax_faltantes, ax_problemas) = nueva_figura(
            1, 2, (13, 5), proporciones=[2.2, 1]
        )

        columnas = [c for c in antes.faltantes
                    if antes.faltantes[c] or despues.faltantes.get(c, 0)]
        columnas += [c for c in despues.faltantes if despues.faltantes[c] and c not in columnas]
        if columnas:
            faltantes = pd.DataFrame(
                [{"columna": c, "momento": "Antes", "faltantes": antes.faltantes.get(c, 0)}
                 for c in columnas]
                + [{"columna": c, "momento": "Después", "faltantes": despues.faltantes.get(c, 0)}
                   for c in columnas]
            )
            _barras(ax_faltantes, faltantes, "columna", "faltantes", columnas)
            ax_faltantes.legend(title=None, frameon=False)
            ax_faltantes.tick_params(axis="x", rotation=20)
        else:
            ax_faltantes.text(0.5, 0.5, "Sin faltantes", ha="center", va="center",
                              color=TINTA_SECUNDARIA, transform=ax_faltantes.transAxes)
            ax_faltantes.set_xticks([])
            ax_faltantes.set_yticks([])
            ax_faltantes.grid(False)
        ax_faltantes.set_xlabel("")
        ax_faltantes.set_ylabel("Valores faltantes")
        titulo(ax_faltantes, "Faltantes por columna")

        categorias = {"Duplicados": (antes.duplicados, despues.duplicados)}
        if antes.edades_atipicas is not None:
            categorias["Edades atípicas"] = (antes.edades_atipicas, despues.edades_atipicas or 0)
        problemas = pd.DataFrame(
            [{"problema": nombre, "momento": momento, "cantidad": valores[i]}
             for nombre, valores in categorias.items()
             for i, momento in enumerate(ORDEN_MOMENTO)]
        )
        _barras(ax_problemas, problemas, "problema", "cantidad", list(categorias))
        if columnas:
            ax_problemas.get_legend().remove()  # la leyenda ya está en el primer panel
        else:
            ax_problemas.legend(title=None, frameon=False)
        ax_problemas.set_xlabel("")
        ax_problemas.set_ylabel("Filas")
        titulo(ax_problemas, "Duplicados y atípicos")

        texto = f"Calidad de datos: {fuente}"
        figura.suptitle(texto, color=TINTA, x=0.02, ha="left", fontsize=13)

        total_antes = sum(antes.faltantes.values())
        total_despues = sum(despues.faltantes.values())
        duplicados_eliminados = antes.duplicados - despues.duplicados
        atipicas = antes.edades_atipicas or 0
        partes = []
        if total_antes == total_despues == 0 and duplicados_eliminados == 0 and atipicas == 0:
            partes.append(f"En {fuente} no se encontraron problemas en las {antes.filas} filas "
                          "revisadas: 0 faltantes, 0 duplicados y ninguna edad atípica.")
        else:
            if total_antes or total_despues:
                partes.append(
                    f"En {fuente} había {total_antes} valores faltantes antes de limpiar y "
                    f"{total_despues} después."
                )
                if total_despues > total_antes and atipicas:
                    partes.append(
                        f"Aumentan porque las {atipicas} edades atípicas se convierten en \"ND\": "
                        "un valor imposible se marca como desconocido en lugar de usarse."
                    )
                elif total_despues < total_antes:
                    partes.append(
                        f"Se resolvieron {total_antes - total_despues} (por ejemplo, la moneda "
                        "vacía se completa como MXN)."
                    )
                partes.append(
                    "Los faltantes que quedan son \"ND\": se conservan como faltantes en lugar de "
                    "rellenarse con 0, y se imputan después con la mediana del conjunto de "
                    "entrenamiento."
                )
            if duplicados_eliminados:
                partes.append(f"Se eliminaron {duplicados_eliminados} filas duplicadas.")
            if atipicas:
                partes.append(f"{atipicas} edades atípicas (fuera de 18 a 70 años) pasaron a "
                              "\"ND\" sin eliminar al cliente.")
        return GraficaInterpretada(f"calidad_{fuente}", texto, figura, " ".join(partes),
                                   f"calidad_{fuente}.png")
