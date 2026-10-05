"""Resultado de la revisión de estructura y calidad de una fuente de datos.

`ReporteCalidad` es lo que devuelve `DataValidator.revisar_calidad` (ver
`specs/006-datavalidator-calidad-datos/data-model.md`). Sirve para comparar
el estado de los datos antes y después de limpiarlos.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ReporteCalidad:
    """Resumen de la estructura y calidad de una fuente de datos.

    Attributes:
        fuente: Nombre de la fuente revisada (p. ej. `"perfil_solicitante"`).
        filas: Número de filas de la tabla.
        columnas_faltantes: Columnas esperadas que no están en la tabla.
        columnas_adicionales: Columnas presentes que no son esperadas.
        tipos: Tipo de dato de cada columna presente.
        faltantes: Cantidad de valores vacíos/`NaN` ("ND") por columna.
        duplicados: Cantidad de filas idénticas a una fila anterior, tal
            como llegan (sin normalizar).
        ids_repetidos: Cantidad de `ID_Cliente` repetidos. `None` en el
            historial transaccional (donde un cliente puede tener varias
            filas) o si falta la columna.
        edades_atipicas: Cantidad de edades numéricas fuera del rango
            válido. `None` fuera del perfil del solicitante o si falta la
            columna `Edad`.
    """

    fuente: str
    filas: int
    columnas_faltantes: list[str]
    columnas_adicionales: list[str]
    tipos: dict[str, str]
    faltantes: dict[str, int]
    duplicados: int
    ids_repetidos: int | None
    edades_atipicas: int | None

    @property
    def estructura_valida(self) -> bool:
        """`True` si la tabla tiene todas sus columnas esperadas."""
        return not self.columnas_faltantes

    def como_tabla(self) -> pd.DataFrame:
        """Devuelve el detalle por columna, útil para mostrarlo en un notebook.

        Returns:
            Un DataFrame con una fila por columna presente y las columnas
            `columna`, `tipo` y `faltantes`.
        """
        return pd.DataFrame(
            {
                "columna": list(self.tipos),
                "tipo": list(self.tipos.values()),
                "faltantes": [self.faltantes[columna] for columna in self.tipos],
            }
        )

    def resumen(self) -> str:
        """Devuelve el reporte como texto legible, listo para imprimir.

        Omite los indicadores que no aplican a la fuente (p. ej. edades
        atípicas en el historial) y solo lista las columnas con faltantes.

        Returns:
            Un texto de varias líneas, por ejemplo::

                Reporte de calidad: perfil_solicitante
                Filas: 138 | Estructura válida: sí
                Duplicados: 3 | IDs repetidos: 3 | Edades atípicas: 9
                Faltantes: Ingreso_Declarado: 5, Antiguedad_Laboral: 7
        """
        estructura = "sí" if self.estructura_valida else "no"
        lineas = [
            f"Reporte de calidad: {self.fuente}",
            f"Filas: {self.filas} | Estructura válida: {estructura}",
        ]
        if self.columnas_faltantes:
            lineas.append(f"Columnas faltantes: {', '.join(self.columnas_faltantes)}")
        if self.columnas_adicionales:
            lineas.append(f"Columnas adicionales: {', '.join(self.columnas_adicionales)}")

        problemas = [f"Duplicados: {self.duplicados}"]
        if self.ids_repetidos is not None:
            problemas.append(f"IDs repetidos: {self.ids_repetidos}")
        if self.edades_atipicas is not None:
            problemas.append(f"Edades atípicas: {self.edades_atipicas}")
        lineas.append(" | ".join(problemas))

        con_faltantes = [f"{columna}: {n}" for columna, n in self.faltantes.items() if n]
        lineas.append("Faltantes: " + (", ".join(con_faltantes) if con_faltantes else "ninguno"))
        return "\n".join(lineas)
