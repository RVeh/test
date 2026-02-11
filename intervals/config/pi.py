from dataclasses import dataclass

@dataclass(frozen=True)
class PIModelConfig:
    """
    Model parameters for prediction intervals.
    """
    p: float
    n: int
    gamma: float = 0.95


@dataclass(frozen=True)
class PIGeometryConfig:
    """
    Geometric parameters for PI plots.
    """
    p_min: float = 0.0
    p_max: float = 1.0
