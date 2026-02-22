"""
tests.py

Referenzprogramme für Hypothesentests
(Sek II / Lehrkräftefortbildung)

Ziel:
- stabile, grafische Referenzwerkzeuge
- Fokus auf Ablehnungsbereich und Entscheidung
- eine Funktion = eine Grafik
- gleiche Struktur wie intervalle.py
"""

# ============================================================
# 0. Imports (einmal, zentral)
# ============================================================

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================
# 1. KONFIGURATIONEN – das dürfen LuL verändern
# ============================================================

# ---------- Modell (Hypothesentest) ----------

@dataclass(frozen=True)
class TestModel:
    """
    Modell für einen z-Test (Anteilstest)

    p0     Wahrscheinlichkeit unter H0
    h      beobachteter Anteil
    n      Stichprobengröße
    alpha  Signifikanzniveau
    """
    p0: float
    h: float
    n: int
    alpha: float = 0.05


# ---------- Geometrie (Darstellungsraum) ----------

@dataclass(frozen=True)
class Geometry:
    """
    Zeichenraum der Grafik
    """
    x_min: float = -4.0
    x_max: float = 4.0
    points: int = 1200


# ---------- Darstellung ----------

@dataclass(frozen=True)
class TestStyle:
    curve: str = "black"
    reject_color: str = "lightcoral"
    accept_color: str = "lightgray"
    statistic_color: str = "tab:blue"

    grid: bool = True
    figsize: tuple = (4, 4)

# ============================================================
# 2. RECHENKERN – Mathematik (Finger weg)
# ============================================================

def z_statistic(h: float, p0: float, n: int) -> float:
    """
    Teststatistik für den Anteilstest
    """
    return (h - p0) / sqrt(p0 * (1 - p0) / n)


def critical_value(alpha: float) -> float:
    """
    Kritischer z-Wert (zweiseitiger Test)
    """
    return NormalDist().inv_cdf(1 - alpha / 2)

# ============================================================
# 3. PLOT-FUNKTION – eine Funktion = eine Grafik
# ============================================================

def plot_z_test(
    model: TestModel,
    geometry: Geometry = Geometry(),
    style: TestStyle = TestStyle(),
    *,
    show_info: bool = True,
    save: str | None = None,
):
    """
    Grafische Darstellung eines z-Tests
    (Ablehnungsbereich und Teststatistik)
    """

    x = np.linspace(geometry.x_min, geometry.x_max, geometry.points)
    y = norm.pdf(x)

    z_obs = z_statistic(model.h, model.p0, model.n)
    z_crit = critical_value(model.alpha)

    fig, ax = plt.subplots(figsize=style.figsize)

    # Dichte
    ax.plot(x, y, color=style.curve)

    # Ablehnungsbereiche
    ax.fill_between(
        x, y, where=(x <= -z_crit),
        color=style.reject_color, alpha=0.8
    )
    ax.fill_between(
        x, y, where=(x >= z_crit),
        color=style.reject_color, alpha=0.8
    )

    # Annahmebereich (optisch ruhig)
    ax.fill_between(
        x, y, where=(abs(x) < z_crit),
        color=style.accept_color, alpha=0.6
    )

    # Teststatistik
    ax.axvline(z_obs, color=style.statistic_color, linewidth=2)

    # Achsen & Titel
    ax.set_xlim(geometry.x_min, geometry.x_max)
    ax.set_yticks([])
    ax.set_xlabel(r"$z$")
    ax.set_title("z-Test – Ablehnungsbereich")

    ax.text(
        0.5, 1.02,
        rf"$H_0: p={model.p0},\; n={model.n},\; \alpha={model.alpha}$",
        transform=ax.transAxes,
        ha="center"
    )

    if show_info:
        decision = (
            "H₀ ablehnen" if abs(z_obs) >= z_crit else "H₀ nicht ablehnen"
        )
        ax.text(
            0.5, -0.15,
            rf"$z_{{obs}}={z_obs:.2f}$,  $z_{{krit}}={z_crit:.2f}$  →  {decision}",
            transform=ax.transAxes,
            ha="center"
        )

    if style.grid:
        ax.grid(True, alpha=0.6)

    if save:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)