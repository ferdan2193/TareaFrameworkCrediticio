"""Componente DataValidator del framework de análisis crediticio.

Expone `DataValidator` para validar la estructura, limpiar y normalizar los
`DataFrame` que produce `DataLoader`, y `ReporteCalidad`, el resultado de su
revisión de estructura y calidad (ver `specs/002-datavalidator-limpieza/` y
`specs/006-datavalidator-calidad-datos/`).
"""

from .reporte import ReporteCalidad
from .validator import DataValidator

__all__ = ["DataValidator", "ReporteCalidad"]
