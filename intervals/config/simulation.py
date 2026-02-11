from dataclasses import dataclass


@dataclass(frozen=True)
class CISimModelConfig:
    """
    Model assumptions for CI simulation.
    """
    n: int
    p_true: float
    gamma: float = 0.95


@dataclass(frozen=True)
class CISimControlConfig:
    """
    Control parameters for Monte Carlo simulation.
    """
    m: int = 100
    seed: int = 42


@dataclass(frozen=True)
class CISimGeometryConfig:
    """
    Geometric / axis parameters for simulation plots.
    """
    x_min: float = 0.0
    x_max: float = 1.0

