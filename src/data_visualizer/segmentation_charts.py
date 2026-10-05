"""Gráficas de la segmentación K-means: silhouette y mapa PCA.

Ver `specs/011-datavisualizer-seaborn/`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from model_evaluator import EvaluationResult
    from model_trainer import TrainingResult

from .estilo import (
    ATENUADO,
    AZUL,
    COLORES_SEGMENTO,
    SUPERFICIE,
    TINTA,
    TINTA_SECUNDARIA,
    nueva_figura,
    titulo,
)
from .grafica import GraficaInterpretada


def _estructura(silhouette: float) -> str:
    if silhouette >= 0.5:
        return "fuerte"
    if silhouette >= 0.25:
        return "moderada"
    return "débil"


class SegmentationCharts:
    """Muestra si los segmentos de clientes son razonables y qué tan separados están."""

    @staticmethod
    def silhouette(evaluacion: EvaluationResult) -> GraficaInterpretada:
        """Silhouette para cada número de segmentos, resaltando el del modelo.

        Args:
            evaluacion: Resultado de `ModelEvaluator.evaluate`.

        Returns:
            Una barra por número de segmentos.
        """
        tabla = evaluacion.silhouette
        figura, ax = nueva_figura(tamano=(7, 5))
        colores = [AZUL if es_el_modelo else ATENUADO for es_el_modelo in tabla["es_el_modelo"]]
        ax.bar(tabla["k"].astype(str), tabla["silhouette"], color=colores, width=0.6)
        for posicion, (k, valor) in enumerate(zip(tabla["k"], tabla["silhouette"])):
            ax.text(posicion, valor, f"{valor:.2f}", ha="center", va="bottom",
                    color=TINTA_SECUNDARIA, fontsize=10)
            if k == evaluacion.k_sugerido:
                ax.text(posicion, valor + 0.025, "Sugerido", ha="center", va="bottom",
                        color=TINTA, fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(0.5, float(tabla["silhouette"].max()) + 0.08))
        ax.set_xlabel("Número de segmentos (k)")
        ax.set_ylabel("Silhouette")
        ax.grid(axis="x", visible=False)
        texto = "Silhouette por número de segmentos"
        titulo(ax, texto)

        k_modelo = int(tabla.loc[tabla["es_el_modelo"], "k"].iloc[0]) if tabla["es_el_modelo"].any() else None
        valor = evaluacion.silhouette_modelo
        interpretacion = (
            f"Con {k_modelo} segmentos, el silhouette es {valor:.2f}: una estructura de grupos "
            f"{_estructura(valor)} (fuerte ≥ 0.50, moderada ≥ 0.25, débil < 0.25). El número con "
            f"mayor silhouette es k = {evaluacion.k_sugerido}"
            + (", el mismo que usa el modelo." if evaluacion.k_sugerido == k_modelo
               else f", distinto de los {k_modelo} que usa el modelo.")
        )
        if _estructura(valor) == "débil":
            interpretacion += (" Los segmentos sirven para describir perfiles de clientes, pero no "
                               "son grupos naturalmente separados.")
        return GraficaInterpretada("silhouette", texto, figura, interpretacion, "silhouette.png")

    @staticmethod
    def pca_segmentos(resultado: TrainingResult) -> GraficaInterpretada:
        """Mapa 2D de los clientes (PCA) coloreado por segmento.

        El PCA se usa solo para visualizar: resume las variables preparadas en
        2 componentes y no cambia el modelo ni la segmentación.

        Args:
            resultado: Resultado de `ModelTrainer.train`.

        Returns:
            Un punto por cliente (entrenamiento y prueba), con una etiqueta
            directa en el centro de cada segmento.
        """
        datos = resultado.datos
        X = pd.concat([datos.X_train, datos.X_test], ignore_index=True)
        pca = PCA(n_components=2, random_state=42)
        componentes = pca.fit_transform(X)
        p1, p2 = pca.explained_variance_ratio_
        segmentos = resultado.segmentos.loc[datos.ids_train + datos.ids_test].to_numpy()
        etiquetas = [f"Segmento {s}" for s in segmentos]
        orden = [f"Segmento {i}" for i in range(resultado.segmentacion.n_segmentos)]
        mapa = pd.DataFrame({"c1": componentes[:, 0], "c2": componentes[:, 1], "segmento": etiquetas})

        figura, ax = nueva_figura(tamano=(8, 6))
        sns.scatterplot(data=mapa, x="c1", y="c2", hue="segmento", hue_order=orden,
                        palette=dict(zip(orden, COLORES_SEGMENTO)), s=50,
                        edgecolor=SUPERFICIE, linewidth=1, ax=ax)
        for nombre, grupo in mapa.groupby("segmento"):
            ax.text(grupo["c1"].mean(), grupo["c2"].mean(), nombre, ha="center", va="center",
                    color=TINTA, fontsize=10, fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": SUPERFICIE,
                          "edgecolor": "none", "alpha": 0.85})
        ax.legend(title=None, frameon=False)
        ax.set_xlabel(f"Componente 1 ({p1:.0%} de la variación)")
        ax.set_ylabel(f"Componente 2 ({p2:.0%} de la variación)")
        texto = "Segmentos de clientes (proyección PCA)"
        titulo(ax, texto)

        total = p1 + p2
        interpretacion = (
            f"El mapa resume las {X.shape[1]} variables en 2 componentes que conservan el "
            f"{total:.0%} de la variación de los datos ({len(mapa)} clientes)."
        )
        if total < 0.5:
            interpretacion += (" Como es menos de la mitad, el mapa simplifica mucho: grupos que "
                               "se ven encimados aquí pueden estar algo más separados en las "
                               "variables originales.")
        conteos = mapa["segmento"].value_counts().reindex(orden, fill_value=0)
        interpretacion += " Tamaños: " + ", ".join(f"{n} = {c} clientes" for n, c in conteos.items()) + "."
        return GraficaInterpretada("pca_segmentos", texto, figura, interpretacion,
                                   "pca_segmentos.png")
