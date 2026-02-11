# tests/model/binomial.py
from __future__ import annotations

from dataclasses import dataclass
from math import comb

from .base_model import FiniteDiscreteModel


@dataclass(frozen=True)
class BinomialModel:
    """
    Binomialmodell X ~ Bin(n, p).

    Reines Modellobjekt:
    - kennt n und p
    - erzeugt daraus ein diskretes Modell (support + pmf)

    Keine Testlogik.
    """

    n: int
    p: float

    def __post_init__(self):
        if not (0 <= self.p <= 1):
            raise ValueError("p muss in [0,1] liegen")
        if self.n <= 0:
            raise ValueError("n muss positiv sein")

        support = list(range(self.n + 1))

        def pmf_fn(k: int) -> float:
            if k < 0 or k > self.n:
                return 0.0
            return comb(self.n, k) * (self.p ** k) * ((1 - self.p) ** (self.n - k))

        object.__setattr__(
            self,
            "_model",
            FiniteDiscreteModel(support=support, pmf_fn=pmf_fn),
        )

    # ---- Weitergabe der Modell-Schnittstelle ----

    @property
    def support(self):
        return self._model.support

    def pmf(self, k: int) -> float:
        return self._model.pmf(k)

    def cdf(self, k: int) -> float:
        return self._model.cdf(k)

    def sf(self, k: int) -> float:
        return self._model.sf(k)
