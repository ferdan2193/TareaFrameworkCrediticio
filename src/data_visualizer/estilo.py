"""Colores y estilo común de las gráficas analíticas.

Los colores vienen de la paleta de referencia de la guía de visualización y se
validaron para daltonismo (ver `specs/011-datavisualizer-seaborn/research.md`,
R3). Reglas:

- Cada entidad tiene siempre el mismo color: la regresión logística es azul y el
  random forest es naranja en todas las gráficas.
- El texto va en tinta, nunca en el color de la serie.
- Las figuras se crean sin `pyplot`, así que no abren ventanas ni acumulan
  estado global.
"""

from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

AZUL = "#2a78d6"
NARANJA = "#eb6834"
AQUA = "#1baf7a"
ROJO = "#e34948"
ATENUADO = "#c3c2b7"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
EJES = "#898781"
REJILLA = "#e1e0d9"
SUPERFICIE = "#fcfcfb"

COLOR_MODELO = {"Regresión logística": AZUL, "Random forest": NARANJA}
"""Color fijo de cada modelo supervisado."""

COLORES_SEGMENTO = [AZUL, NARANJA, AQUA]
"""Colores de los segmentos de K-means (validados con todos los pares para el scatter)."""

MAPA_SECUENCIAL = LinearSegmentedColormap.from_list(
    "azul_secuencial", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
)
"""Un solo tono, de claro (poco) a oscuro (mucho). Se usa en la matriz de confusión."""

MAPA_DIVERGENTE = LinearSegmentedColormap.from_list(
    "azul_gris_rojo", [AZUL, "#f0efec", ROJO]
)
"""Azul (negativo) ↔ gris (cero) ↔ rojo (positivo). Se usa en las correlaciones."""


def _aplicar_estilo(ax: Axes) -> None:
    ax.set_facecolor(SUPERFICIE)
    ax.set_axisbelow(True)
    ax.grid(True, color=REJILLA, linewidth=0.8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EJES)
    ax.tick_params(colors=EJES, labelcolor=TINTA_SECUNDARIA)
    ax.xaxis.label.set_color(TINTA_SECUNDARIA)
    ax.yaxis.label.set_color(TINTA_SECUNDARIA)


def titulo(ax: Axes, texto: str) -> None:
    """Pone el título de un panel en tinta, alineado a la izquierda."""
    ax.set_title(texto, color=TINTA, loc="left", fontsize=12)


def nueva_figura(
    filas: int = 1,
    columnas: int = 1,
    tamano: tuple[float, float] = (8, 5),
    proporciones: list[float] | None = None,
):
    """Crea una figura con el estilo del proyecto, sin usar `pyplot`.

    Args:
        filas: Número de filas de paneles.
        columnas: Número de columnas de paneles.
        tamano: Tamaño de la figura en pulgadas.
        proporciones: Ancho relativo de cada columna de paneles (opcional).

    Returns:
        `(figura, ejes)`. `ejes` es un solo `Axes` si hay un panel, o un
        arreglo de `Axes` si hay varios.
    """
    figura = Figure(figsize=tamano, facecolor=SUPERFICIE, layout="constrained")
    ejes = figura.subplots(
        filas, columnas, squeeze=True,
        gridspec_kw={"width_ratios": proporciones} if proporciones else None,
    )
    for ax in figura.axes:
        _aplicar_estilo(ax)
    return figura, ejes
