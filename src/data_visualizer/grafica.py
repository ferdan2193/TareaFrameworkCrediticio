"""Gráfica con su interpretación escrita.

Ver `specs/011-datavisualizer-seaborn/data-model.md`.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

from matplotlib.figure import Figure

from .estilo import SUPERFICIE


@dataclass(frozen=True)
class GraficaInterpretada:
    """Una gráfica lista para el reporte, con su interpretación.

    Attributes:
        identificador: Nombre único de la gráfica, p. ej. `"matriz_confusion"`.
        titulo: Título de la gráfica.
        figura: La figura de matplotlib.
        interpretacion: Texto en español generado a partir de los mismos datos
            que se dibujan. Es una base para el reporte; el análisis personal
            lo redacta el estudiante.
        archivo: Nombre del archivo PNG, sin carpeta.
        ruta: Ruta completa una vez guardada (`None` antes de guardar).
    """

    identificador: str
    titulo: str
    figura: Figure = field(compare=False, repr=False)
    interpretacion: str
    archivo: str
    ruta: Path | None = None

    def guardar(self, carpeta: str | Path) -> "GraficaInterpretada":
        """Guarda la figura como PNG en `carpeta`, que se crea si no existe.

        Args:
            carpeta: Carpeta de destino.

        Returns:
            Una copia de esta gráfica con `ruta` llenada.
        """
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / self.archivo
        self.figura.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=SUPERFICIE)
        return dataclasses.replace(self, ruta=ruta)
