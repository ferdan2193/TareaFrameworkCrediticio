"""Comparación de los modelos supervisados y recomendación de uno.

Ver `specs/010-model-evaluator/` (criterio acordado en la sección Clarifications).
"""

from dataclasses import dataclass

import pandas as pd

from .metrics import MetricasModelo

METRICAS = ["exactitud", "precision", "recall", "f1"]
MODELO_MAS_SIMPLE = "Regresión logística"
UMBRAL_RECALL_BAJO = 0.5
_EMPATE = 1e-9


@dataclass(frozen=True)
class Recomendacion:
    """Modelo recomendado y por qué.

    Attributes:
        modelo: Nombre del modelo recomendado.
        criterio: Lo que decidió: `"precision_cv"`, `"f1_cv"`, `"simplicidad"`
            o `"unico_modelo"`.
        diferencia_significativa: `False` si la diferencia de precisión con el
            segundo modelo es menor que la desviación de la validación cruzada,
            es decir, si la elección es débil.
        justificacion: Explicación en español, lista para el reporte.
    """

    modelo: str
    criterio: str
    diferencia_significativa: bool
    justificacion: str


class ModelComparator:
    """Compara los modelos lado a lado y recomienda uno."""

    @staticmethod
    def comparar(
        metricas: dict[str, MetricasModelo], cv: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Arma la tabla comparativa.

        Args:
            metricas: Métricas de prueba por modelo.
            cv: Tabla de validación cruzada por modelo (de `CrossValidator`).

        Returns:
            Un DataFrame con una fila por modelo y columnas
            `<métrica>_prueba`, `auc_prueba`, `<métrica>_cv` y `<métrica>_cv_std`.
        """
        filas = {}
        for nombre, m in metricas.items():
            fila = {f"{metrica}_prueba": getattr(m, metrica) for metrica in METRICAS}
            fila["auc_prueba"] = m.auc_roc
            for metrica in METRICAS:
                fila[f"{metrica}_cv"] = cv[nombre].loc[metrica, "media"]
                fila[f"{metrica}_cv_std"] = cv[nombre].loc[metrica, "desviacion"]
            filas[nombre] = fila
        return pd.DataFrame.from_dict(filas, orient="index")

    @staticmethod
    def recomendar(tabla: pd.DataFrame) -> Recomendacion:
        """Recomienda el modelo con mayor precisión en validación cruzada.

        Criterio (acordado con el usuario): mayor `precision_cv`, porque
        aprobar a quien no cumple la política es el error más costoso. Si hay
        empate, mayor `f1_cv`. Si persiste, la regresión logística, por ser
        más interpretable.

        Args:
            tabla: Resultado de `comparar`.

        Returns:
            Una `Recomendacion` con el modelo, el criterio que decidió, si la
            diferencia es significativa y una justificación en español.
        """
        orden = tabla.assign(
            _simple=[nombre == MODELO_MAS_SIMPLE for nombre in tabla.index]
        ).sort_values(["precision_cv", "f1_cv", "_simple"], ascending=False, kind="stable")
        elegido = orden.index[0]
        fila = orden.iloc[0]

        if len(orden) == 1:
            criterio, significativa, segundo = "unico_modelo", False, None
        else:
            segundo = orden.iloc[1]
            if abs(fila["precision_cv"] - segundo["precision_cv"]) > _EMPATE:
                criterio = "precision_cv"
            elif abs(fila["f1_cv"] - segundo["f1_cv"]) > _EMPATE:
                criterio = "f1_cv"
            else:
                criterio = "simplicidad"
            diferencia = abs(fila["precision_cv"] - segundo["precision_cv"])
            significativa = bool(
                diferencia >= max(fila["precision_cv_std"], segundo["precision_cv_std"])
            )

        partes = [
            f"Se recomienda {elegido}, con una precisión media de {fila['precision_cv']:.3f} "
            "en validación cruzada. La precisión es el criterio principal porque, cuando el "
            "modelo aprueba un crédito, debe acertar: aprobar a quien no cumple la política "
            "(falso positivo) es el error más costoso."
        ]
        if criterio == "f1_cv":
            partes.append(
                f"La precisión empató, así que decidió el F1 ({fila['f1_cv']:.3f} contra "
                f"{segundo['f1_cv']:.3f})."
            )
        elif criterio == "simplicidad":
            partes.append(
                "Precisión y F1 empataron, así que se prefiere el modelo más simple e "
                "interpretable."
            )
        if segundo is not None and not significativa:
            partes.append(
                f"Sin embargo, la diferencia con {orden.index[1]} "
                f"({fila['precision_cv']:.3f} contra {segundo['precision_cv']:.3f}) no es "
                "significativa: es menor que la variación entre particiones "
                f"(desviación de hasta {max(fila['precision_cv_std'], segundo['precision_cv_std']):.3f}). "
                "Con estos datos, ambos modelos son prácticamente equivalentes."
            )
        if fila["recall_cv"] < UMBRAL_RECALL_BAJO:
            partes.append(
                f"Advertencia: su recall en validación cruzada es {fila['recall_cv']:.3f}, es decir, "
                "rechaza a más de la mitad de los clientes que sí cumplen la política."
            )

        return Recomendacion(
            modelo=elegido,
            criterio=criterio,
            diferencia_significativa=significativa,
            justificacion=" ".join(partes),
        )
