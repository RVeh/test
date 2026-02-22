"""
plot_style.py

Einheitliche, didaktische Plot-„Typografie“ für Referenzgrafiken.

Ziel:
- konsistente Titel/Subtitel/Infozeilen (zentriert, achsenrelativ)
- ruhiges Layout (für Unterricht + Beamer/PDF)
- Binder-stabil (keine globalen Side-Effects)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TextBlockStyle:
    title_fs: int = 12
    subtitle_fs: int = 10
    info_fs: int = 10

    # relative Positionen in Achsenkoordinaten
    title_y: float = 1.07
    subtitle_y: float = 1.02
    info_y: float = -0.15


def centered_text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: int,
    va: str = "bottom",
):
    """Zentrierter Text in Achsenkoordinaten (transform=ax.transAxes)."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va=va,
        fontsize=fontsize,
    )


def set_title_block(
    ax,
    title: str,
    subtitle: Optional[str] = None,
    *,
    style: TextBlockStyle = TextBlockStyle(),
):
    """Einheitlicher Titel (und optional Subtitel) oberhalb der Achse."""
    centered_text(ax, 0.5, style.title_y, title, fontsize=style.title_fs, va="bottom")
    if subtitle:
        centered_text(ax, 0.5, style.subtitle_y, subtitle, fontsize=style.subtitle_fs, va="bottom")


def set_info_line(
    ax,
    text: str,
    *,
    style: TextBlockStyle = TextBlockStyle(),
):
    """Einheitliche Infozeile unterhalb der Achse."""
    centered_text(ax, 0.5, style.info_y, text, fontsize=style.info_fs, va="top")


def apply_grid(ax, *, enabled: bool = True, alpha: float = 0.8):
    """Ruhiges Standardgrid."""
    if enabled:
        ax.grid(True, alpha=alpha)


def finalize_figure(
    fig,
    *,
    save: str | None = None,
    show: bool = True,
):
    """
    Einheitlicher Abschluss:
    - optional speichern (bbox_inches='tight')
    - optional anzeigen
    - immer schließen (Binder-stabil)
    """
    if save:
        fig.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
