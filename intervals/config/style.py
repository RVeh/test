from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class CIStyle:
    curve_upper: str = "black"
    curve_lower: str = "green"
    ci_bar: str = "blue"
    helper_lines: str = "gray"

    area_color: str = "lightgray"
    area_alpha: float = 0.4   # 0.0 → aus
    
    grid: bool = True
    ticks: str = "normal"   # "normal" | "fine"

    figsize: Tuple[int, int] = (4, 4)
    
    show_prediction_overlay: bool = False
    prediction_steps: int = 10
    prediction_alpha: float = 0.8


@dataclass(frozen=True)
class PIStyle:
    curve_upper: str = "black"
    curve_lower: str = "green"
    
    interval_bar: str = "blue"
    helper_lines: str = "gray"

    area_color: str = "lightgray"
    area_alpha: float = 0.0   # standardmäßig aus bei PI

    grid: bool = True
    ticks: str = "normal"   # "normal" | "fine"

    figsize: Tuple[int, int] = (4, 4)


@dataclass(frozen=True)
class CISimStyle:
    """
    Style settings for CI simulation plots.
    """
    color_cover: str = "blue"
    color_miss: str = "red"

    figsize: Tuple[float, float] = (4.2, 6.2)

    show_stats: bool = True
