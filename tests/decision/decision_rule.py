# tests/decision/decision_rule.py
from __future__ import annotations

from dataclasses import dataclass

from tests.geometry.rejection_region import RejectionRegion


@dataclass(frozen=True)
class TestDecision:
    """
    Ergebnis eines Hypothesentests.

    Keine Interpretation, keine Statistik:
    - reject: True / False
    - x_obs : beobachteter Wert
    """
    x_obs: int
    reject: bool

    def __str__(self) -> str:
        if self.reject:
            return f"H0 wird verworfen (x = {self.x_obs} ∈ K)"
        return f"H0 wird nicht verworfen (x = {self.x_obs} ∉ K)"


def decision_rule(x_obs: int, rejection_region: RejectionRegion) -> TestDecision:
    """
    Entscheidungsregel eines Tests:

    Verwerfe H0 genau dann, wenn
    die Beobachtung im Ablehnungsbereich liegt.
    """
    reject = rejection_region.contains(x_obs)
    return TestDecision(x_obs=x_obs, reject=reject)
