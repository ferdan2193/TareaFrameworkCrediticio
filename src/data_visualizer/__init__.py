"""Componente DataVisualizer del framework de análisis crediticio.

Expone `DataVisualizer` como única clase pública para generar
visualizaciones exploratorias de los `DataFrame` que produce
`DataValidator` (ver `specs/003-datavisualizer-graficas/`).
"""

from .visualizer import DataVisualizer

__all__ = ["DataVisualizer"]
