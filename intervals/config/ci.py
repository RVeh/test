from dataclasses import dataclass

@dataclass(frozen=True)

class CIModelConfig:
    """
    Model parameters for confidence intervals.
    """
    h: float
    n: int
    gamma: float

@dataclass(frozen=True)
class CIGeometryConfig:
    """
    Geometric parameters for CI plots.
    """
    p_min: float = 0.0
    p_max: float = 0.5
    points: int = 3000
