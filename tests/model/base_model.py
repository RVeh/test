# tests/model/base_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Protocol, runtime_checkable, Any


@runtime_checkable
class DiscreteModel(Protocol):
    """
    Diskretes Wahrscheinlichkeitsmodell (für Tests i.d.R. unter H0).

    Ein Modell liefert NUR:
    - den Ergebnisraum (support)
    - eine Wahrscheinlichkeitsfunktion (pmf)
    - optional: Erwartungswert / Varianz (für Geometrie/Erklärungen, nicht für Entscheidungen)

    Keine Testlogik, kein Ablehnungsbereich, kein p-Wert.
    """

    @property
    def support(self) -> Sequence[int]:
        ...

    def pmf(self, x: int) -> float:
        ...

    def cdf(self, x: int) -> float:
        ...

    def sf(self, x: int) -> float:
        ...


@dataclass(frozen=True)
class FiniteDiscreteModel:
    """
    Konkrete, generische Implementierung für endliche diskrete Modelle.

    Du gibst:
    - support: Liste der möglichen Werte
    - pmf_fn : Funktion x -> P(X=x)

    Vorteil:
    - sehr ehrlich: Modell ist explizit ein Objekt, nicht nur 'n und p'
    - super für Unterricht/Didaktik: alles sichtbar
    """
    support: Sequence[int]
    pmf_fn: Callable[[int], float]

    def pmf(self, x: int) -> float:
        return float(self.pmf_fn(x))

    def cdf(self, x: int) -> float:
        # P(X <= x)
        s = 0.0
        for xi in self.support:
            if xi <= x:
                s += self.pmf(xi)
        return float(s)

    def sf(self, x: int) -> float:
        # P(X >= x)
        s = 0.0
        for xi in self.support:
            if xi >= x:
                s += self.pmf(xi)
        return float(s)
