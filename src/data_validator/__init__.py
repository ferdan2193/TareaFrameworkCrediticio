"""Componente DataValidator del framework de análisis crediticio.

Expone `DataValidator` como única clase pública para limpiar y normalizar
los `DataFrame` que produce `DataLoader` (ver
`specs/002-datavalidator-limpieza/`).
"""

from .validator import DataValidator

__all__ = ["DataValidator"]
