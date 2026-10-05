"""Gráficas exploratorias: distribución de clases y correlaciones.

Ver `specs/011-datavisualizer-seaborn/`.
"""

import pandas as pd
import seaborn as sns

from .estilo import (
    AZUL,
    EJES,
    MAPA_DIVERGENTE,
    NARANJA,
    ROJO,
    SUPERFICIE,
    TINTA_SECUNDARIA,
    nueva_figura,
    titulo,
)
from .grafica import GraficaInterpretada

UMBRAL_DESBALANCE = 0.40
UMBRAL_MODERADA = 0.30
"""|r| a partir del cual una correlación se considera moderada."""
UMBRAL_DEBIL = 0.10
"""|r| por debajo del cual una correlación se considera prácticamente nula."""


def _correlaciones(tabla_por_cliente: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Correlación de Pearson entre las columnas numéricas no constantes.

    Returns:
        `(matriz de correlación, columnas constantes excluidas)`.
    """
    numericas = tabla_por_cliente.drop(columns="ID_Cliente", errors="ignore").select_dtypes("number")
    desviacion = numericas.std()
    constantes = [c for c in numericas.columns if not desviacion[c] > 0]
    return numericas.drop(columns=constantes).corr(), constantes


class ExploratoryCharts:
    """Exploración de los datos antes de modelar."""

    @staticmethod
    def distribucion_clases(tabla_por_cliente: pd.DataFrame) -> GraficaInterpretada:
        """Cuántos clientes quedaron aprobados y cuántos no.

        Args:
            tabla_por_cliente: Tabla etiquetada con una fila por cliente.

        Returns:
            Una barra por clase con su conteo y porcentaje.
        """
        conteos = tabla_por_cliente["credito_aprobado"].value_counts().reindex([0, 1], fill_value=0)
        total = int(conteos.sum())
        datos = pd.DataFrame({"clase": ["No aprobado", "Aprobado"], "clientes": conteos.to_numpy()})

        figura, ax = nueva_figura(tamano=(6, 5))
        sns.barplot(data=datos, x="clase", y="clientes", hue="clase", legend=False,
                    palette={"No aprobado": NARANJA, "Aprobado": AZUL}, ax=ax, width=0.6,
                    saturation=1)
        for barra, clientes in zip(ax.patches, datos["clientes"]):
            ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height(),
                    f"{clientes} ({clientes / total:.0%})", ha="center", va="bottom",
                    color=TINTA_SECUNDARIA, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Clientes")
        ax.grid(axis="x", visible=False)
        texto = "Distribución de credito_aprobado"
        titulo(ax, texto)

        proporcion = conteos[1] / total
        minoritaria = min(proporcion, 1 - proporcion)
        interpretacion = (
            f"De {total} clientes, {conteos[1]} quedaron aprobados ({proporcion:.0%}) y "
            f"{conteos[0]} no aprobados ({1 - proporcion:.0%}). "
        )
        if minoritaria < UMBRAL_DESBALANCE:
            interpretacion += (
                "Hay desbalance de clases: la exactitud sola sería engañosa, por eso se evalúa con "
                "precisión y recall, y la separación en entrenamiento y prueba conserva esta "
                "proporción."
            )
        else:
            interpretacion += "No hay un desbalance fuerte entre las clases."
        return GraficaInterpretada("distribucion_clases", texto, figura, interpretacion,
                                   "distribucion_clases.png")

    @staticmethod
    def correlaciones(tabla_por_cliente: pd.DataFrame) -> GraficaInterpretada:
        """Mapa de calor de correlaciones entre las variables y la aprobación.

        Usa la tabla por cliente en escala original, antes de imputar, con
        correlación de Pearson por pares (ignora los faltantes). Excluye
        `ID_Cliente` y las columnas constantes.

        Args:
            tabla_por_cliente: Tabla etiquetada con una fila por cliente.

        Returns:
            Un mapa de calor con el valor de cada correlación.
        """
        correlacion, constantes = _correlaciones(tabla_por_cliente)

        figura, ax = nueva_figura(tamano=(11, 9))
        sns.heatmap(correlacion, annot=True, fmt=".2f", cmap=MAPA_DIVERGENTE, vmin=-1, vmax=1,
                    center=0, square=True, linewidths=1, linecolor=SUPERFICIE, ax=ax,
                    annot_kws={"fontsize": 8},
                    cbar_kws={"label": "Correlación (Pearson)", "shrink": 0.8})
        ax.grid(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        texto = "Correlaciones entre variables y credito_aprobado"
        titulo(ax, texto)

        con_aprobacion = correlacion["credito_aprobado"].drop("credito_aprobado")
        principales = con_aprobacion.reindex(con_aprobacion.abs().sort_values(ascending=False).index).head(3)
        descripciones = [
            f"{variable} ({valor:+.2f}, {'a mayor valor, más aprobaciones' if valor > 0 else 'a mayor valor, menos aprobaciones'})"
            for variable, valor in principales.items()
        ]
        interpretacion = (
            "Las 3 variables más correlacionadas con la aprobación son: " + "; ".join(descripciones)
            + ". La correlación solo mide relaciones lineales de una variable a la vez; la regla "
            "real combina varias condiciones con umbrales, así que una correlación baja no "
            "significa que la variable no importe."
        )
        if constantes:
            interpretacion += f" Se excluyeron por ser constantes: {', '.join(constantes)}."
        return GraficaInterpretada("correlaciones", texto, figura, interpretacion,
                                   "correlaciones.png")

    @staticmethod
    def correlacion_con_aprobacion(tabla_por_cliente: pd.DataFrame) -> GraficaInterpretada:
        """Barras con la correlación de cada variable con `credito_aprobado`.

        Es la última fila del mapa de correlaciones, ordenada de la más
        positiva a la más negativa para leerla de un vistazo. Rojo = a mayor
        valor, más aprobaciones; azul = menos. Las líneas punteadas marcan
        |r| = 0.3 (relación moderada).

        Args:
            tabla_por_cliente: Tabla etiquetada con una fila por cliente.

        Returns:
            Una barra horizontal por variable con su coeficiente.
        """
        correlacion, constantes = _correlaciones(tabla_por_cliente)
        valores = correlacion["credito_aprobado"].drop("credito_aprobado").dropna().sort_values()

        figura, ax = nueva_figura(tamano=(9, 0.45 * len(valores) + 1.8))
        colores = [ROJO if v > 0 else AZUL for v in valores]
        ax.barh(valores.index, valores.to_numpy(), color=colores, height=0.65)
        ax.axvline(0, color=EJES, linewidth=1)
        for limite in (-UMBRAL_MODERADA, UMBRAL_MODERADA):
            ax.axvline(limite, color=EJES, linewidth=0.8, linestyle="--")
        for posicion, valor in enumerate(valores):
            ax.text(valor + (0.015 if valor >= 0 else -0.015), posicion, f"{valor:+.2f}",
                    va="center", ha="left" if valor >= 0 else "right",
                    color=TINTA_SECUNDARIA, fontsize=9)
        limite_x = max(0.6, float(valores.abs().max()) + 0.12)
        ax.set_xlim(-limite_x, limite_x)
        ax.set_xlabel("Correlación con credito_aprobado (Pearson)")
        ax.set_ylabel("")
        ax.grid(axis="y", visible=False)
        texto = "Correlación de cada variable con credito_aprobado"
        titulo(ax, texto)

        positiva, negativa = valores.idxmax(), valores.idxmin()
        moderadas = valores[valores.abs() >= UMBRAL_MODERADA]
        nulas = valores[valores.abs() < UMBRAL_DEBIL]
        partes = [
            f"La variable con mayor relación positiva es {positiva} ({valores[positiva]:+.2f}: a mayor "
            f"valor, más aprobaciones) y la de mayor relación negativa es {negativa} "
            f"({valores[negativa]:+.2f}: a mayor valor, menos aprobaciones).",
            f"{len(moderadas)} de {len(valores)} variables superan |r| = {UMBRAL_MODERADA:.1f}"
            + (f" ({', '.join(moderadas.index)})" if len(moderadas) else "")
            + f", y {len(nulas)} tienen una correlación prácticamente nula (|r| < {UMBRAL_DEBIL:.1f}).",
            "Una correlación baja no implica que la variable no importe: la regla de aprobación usa "
            "umbrales (edad, score) que una correlación lineal no captura bien.",
        ]
        if constantes:
            partes.append(f"Se excluyeron por ser constantes: {', '.join(constantes)}.")
        return GraficaInterpretada("correlacion_aprobacion", texto, figura, " ".join(partes),
                                   "correlacion_aprobacion.png")
