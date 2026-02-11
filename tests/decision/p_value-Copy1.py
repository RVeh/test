# tests/decision/p_value.py
from __future__ import annotations

from typing import Iterable, Protocol


class DiscreteModel(Protocol):
    @property
    def support(self) -> Iterable[int]:
        ...

    def pmf(self, x: int) -> float:
        ...


def p_value_two_sided_equal_tails(
    model: DiscreteModel,
    x_obs: int,
) -> float:
    """
    Zweiseitiger p-Wert (equal tails, schulische Setzung).

    p = P(X <= x_obs) + P(X >= spiegel(x_obs))
    """
    center = sum(x * model.pmf(x) for x in model.support)
    mirror = int(round(2 * center - x_obs))
    
    #left = sum(model.pmf(x) for x in model.support if x <= x_obs)
    #mirror = int(round(2 * center - x_obs))
    #right = sum(model.pmf(x) for x in model.support if x >= mirror)
    
    p_val = sum(
        model.pmf(x)
        for x in model.support
        if (x <= x_obs) or (x >= mirror)
    )

    # numerische Sicherheit (Rundung / float-Summen)
    return min(1.0, p_val)
    #return left + right


def p_value_symmetric(x_obs: int, model: DiscreteModel, center: float) -> float:
    """
    Alternative p-Wert-Definition:
    Summe aller Wahrscheinlichkeiten,
    die <= pmf(x_obs) sind (klassisch NP-artig).
    """
    px = model.pmf(x_obs)
    return sum(model.pmf(x) for x in model.support if model.pmf(x) <= px)
