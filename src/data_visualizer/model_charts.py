"""Gráficas del desempeño de los modelos: matriz de confusión, ROC e importancia.

Ver `specs/011-datavisualizer-seaborn/`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, roc_curve

if TYPE_CHECKING:
    # Solo para anotaciones: importarlos de verdad crearía un ciclo
    # (preprocessor → data_visualizer → model_evaluator → ... → preprocessor).
    from model_evaluator import EvaluationResult
    from model_trainer import TrainingResult

from .estilo import (
    ATENUADO,
    COLOR_MODELO,
    EJES,
    MAPA_SECUENCIAL,
    SUPERFICIE,
    TINTA,
    nueva_figura,
    titulo,
)
from .grafica import GraficaInterpretada

ETIQUETAS_CLASE = ["No aprobado", "Aprobado"]


class ModelCharts:
    """Gráficas que muestran qué tan bien funcionan los modelos y qué aprendieron."""

    @staticmethod
    def matriz_confusion(evaluacion: EvaluationResult) -> GraficaInterpretada:
        """Matriz de confusión de cada modelo sobre el conjunto de prueba.

        Args:
            evaluacion: Resultado de `ModelEvaluator.evaluate`.

        Returns:
            Un panel por modelo, con el conteo y el porcentaje de cada celda.
        """
        modelos = list(evaluacion.metricas)
        figura, ejes = nueva_figura(1, len(modelos), (5 * len(modelos), 4.5))
        ejes = np.atleast_1d(ejes)

        partes = []
        for ax, nombre in zip(ejes, modelos):
            m = evaluacion.metricas[nombre]
            matriz = np.array(
                [[m.verdaderos_negativos, m.falsos_positivos],
                 [m.falsos_negativos, m.verdaderos_positivos]]
            )
            total = matriz.sum()
            anotaciones = [[f"{v}\n{v / total:.0%}" for v in fila] for fila in matriz]
            sns.heatmap(
                matriz, annot=anotaciones, fmt="", cmap=MAPA_SECUENCIAL, cbar=False,
                xticklabels=ETIQUETAS_CLASE, yticklabels=ETIQUETAS_CLASE,
                linewidths=2, linecolor=SUPERFICIE, square=True, ax=ax,
                annot_kws={"fontsize": 12},
            )
            ax.set_xlabel("Predicho")
            ax.set_ylabel("Real")
            ax.grid(False)
            titulo(ax, nombre)
            partes.append(
                f"{nombre}: {m.verdaderos_positivos} aprobaciones correctas, "
                f"{m.falsos_positivos} aprobaciones indebidas (falsos positivos) y "
                f"{m.falsos_negativos} buenos clientes rechazados (falsos negativos), "
                f"de {total} clientes de prueba."
            )

        menos_fp = min(modelos, key=lambda n: evaluacion.metricas[n].falsos_positivos)
        partes.append(
            f"El modelo con menos aprobaciones indebidas, el error más costoso, es {menos_fp}."
        )
        texto = "Matriz de confusión por modelo"
        figura.suptitle(texto, color=TINTA, x=0.02, ha="left", fontsize=13)
        return GraficaInterpretada("matriz_confusion", texto, figura, " ".join(partes),
                                   "matriz_confusion.png")

    @staticmethod
    def curva_roc(resultado: TrainingResult) -> GraficaInterpretada:
        """Curvas ROC de ambos modelos sobre el conjunto de prueba.

        Args:
            resultado: Resultado de `ModelTrainer.train`.

        Returns:
            Una gráfica con una curva por modelo y la diagonal del azar.
        """
        datos = resultado.datos
        figura, ax = nueva_figura(tamano=(7, 6))
        texto = "Curva ROC: capacidad de separar aprobados de no aprobados"
        titulo(ax, texto)

        auc = {}
        if datos.y_test.nunique() == 2:
            for nombre, modelo in resultado.modelos.items():
                probabilidad = modelo.predict_proba(datos.X_test)
                fpr, tpr, _ = roc_curve(datos.y_test, probabilidad)
                auc[nombre] = float(roc_auc_score(datos.y_test, probabilidad))
                ax.plot(fpr, tpr, color=COLOR_MODELO.get(nombre, EJES), linewidth=2,
                        label=f"{nombre} (AUC {auc[nombre]:.2f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color=EJES, linewidth=1.2, label="Azar (AUC 0.50)")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", frameon=False)

        if not auc:
            interpretacion = (
                "Curva ROC no disponible: el conjunto de prueba tiene una sola clase, así que "
                "no se puede medir cuánto separa cada modelo a los aprobados de los no aprobados."
            )
        else:
            orden = sorted(auc, key=auc.get, reverse=True)
            interpretacion = (
                f"{orden[0]} separa mejor a los clientes que cumplen la política de los que "
                f"no (AUC {auc[orden[0]]:.2f}"
                + (f" contra {auc[orden[1]]:.2f} de {orden[1]}" if len(orden) > 1 else "")
                + "). Un AUC de 0.50 equivale al azar y 1.00 a una separación perfecta; cuanto "
                "más se acerca la curva a la esquina superior izquierda, mejor."
            )
        return GraficaInterpretada("curva_roc", texto, figura, interpretacion, "curva_roc.png")

    @staticmethod
    def importancia(evaluacion: EvaluationResult) -> GraficaInterpretada:
        """Importancia de variables de cada modelo, resaltando las de la regla.

        Args:
            evaluacion: Resultado de `ModelEvaluator.evaluate`.

        Returns:
            Un panel por modelo con barras horizontales ordenadas.
        """
        modelos = list(evaluacion.importancias)
        figura, ejes = nueva_figura(1, len(modelos), (7 * len(modelos), 6))
        ejes = np.atleast_1d(ejes)

        partes = []
        for ax, nombre in zip(ejes, modelos):
            tabla = evaluacion.importancias[nombre]
            es_coeficiente = (tabla["valor"] < 0).any() or nombre == "Regresión logística"
            color = COLOR_MODELO.get(nombre, EJES)
            colores = [color if regla else ATENUADO for regla in tabla["de_la_regla"]]
            posiciones = np.arange(len(tabla))[::-1]  # la más importante arriba
            ax.barh(posiciones, tabla["valor"], color=colores, height=0.7)
            ax.set_yticks(posiciones, tabla["variable"])
            if es_coeficiente:
                ax.axvline(0, color=EJES, linewidth=1)
                ax.set_xlabel("Coeficiente (escala estandarizada)")
            else:
                ax.set_xlabel("Importancia (suma 1)")
            ax.grid(axis="y", visible=False)
            ax.legend(
                handles=[Patch(color=color, label="Variable de la regla"),
                         Patch(color=ATENUADO, label="Otra variable")],
                loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False,
            )
            titulo(ax, nombre)

            principales = tabla.head(3)
            de_la_regla = int(principales["de_la_regla"].sum())
            partes.append(
                f"{nombre}: sus 3 variables más importantes son "
                f"{', '.join(principales['variable'])}; {de_la_regla} de 3 son de la regla de "
                "aprobación."
            )
            if es_coeficiente:
                partes.append(
                    "En la regresión logística, un coeficiente positivo aumenta la probabilidad de "
                    "aprobación y uno negativo la reduce."
                )

        texto = "Importancia de variables por modelo"
        figura.suptitle(texto, color=TINTA, x=0.02, ha="left", fontsize=13)
        return GraficaInterpretada("importancia_variables", texto, figura, " ".join(partes),
                                   "importancia_variables.png")
