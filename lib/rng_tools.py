"""
rng_tools.py

Zentrale Regelung für Zufallszahlen in Simulationen.

Didaktische Leitidee:
- Zufall ist modelliert, nicht dekorativ.
- Reproduzierbarkeit und echte Zufälligkeit sind bewusst unterscheidbar.
"""

from __future__ import annotations
import numpy as np


def get_rng(seed: int | None = 42) -> np.random.Generator:
    """
    Liefert einen numpy-Zufallszahlengenerator.

    seed = 42
        → reproduzierbare Referenzsimulation
        → geeignet für Unterricht, Skripte, Vorträge

    seed = None
        → echte Zufallsrealisierung
        → geeignet, um Schwankungen sichtbar zu machen

    Jeder Simulationsaufruf verwendet genau EINEN Generator.
    """
    return np.random.default_rng(seed)