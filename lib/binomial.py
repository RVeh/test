"""
Binomialverteilung und darauf aufbauende Tests.

Diese Datei enthält:
- das Modell X ~ Bin(n,p)
- Wahrscheinlichkeiten (punktweise, kumulativ, Bereiche)
- grafische Darstellungen
- Testkonstruktionen
- Simulationen

Alle Inhalte beziehen sich auf die Stichprobenperspektive.
Konfidenzintervalle werden bewusst nicht hier behandelt.

Hinweis:
Die Beziehung zwischen Tests und Konfidenzintervallen
wird in den Notebooks hergestellt.
Sie ist bewusst nicht Teil dieser Datei.

"""

# ======= 1. Statistisches Modell =======

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import binom


# ========= Data-Klassen ====================

@dataclass(frozen=True)
class BinomialModel:
    """
    Binomialmodell. Nicht p ist zufällig - sondern das Ergebnis der Stichprobe

    X ~ Bin(n, p)

    n : Anzahl der Versuche
    p : Trefferwahrscheinlichkeit
    """
    n: int
    p: float


@dataclass(frozen=True)
class DistributionPlotStyle:
    """
    Darstellungsstil für Binomialverteilungen.

    sigma_range:
        Bestimmt den sichtbaren Bereich um den Erwartungswert
        in Einheiten der Standardabweichung.
    """
    sigma_range: float = 5.0
    bar_color: str = "tab:blue"
    highlight_color: str = "tab:orange"
    width: float = 0.4 
    grid: bool = True
    figsize: Tuple[float, float] = (8, 4)


@dataclass(frozen=True)
class BinomialTest:
    """
    Binomialtest für die Nullhypothese H0: p = p0.
    """
    n: int
    p0: float
    alpha: float

@dataclass(frozen=True)
class BinomialSimulation:
    """
    Simulation eines Binomialtests.

    n       Stichprobengröße
    p_true  wahrer Parameterwert
    m       Anzahl der Wiederholungen
    seed    Zufallsstartwert
    """
    n: int
    p_true: float
    m: int = 1000
    seed: int | None = 42


@dataclass(frozen=True)
class TypographyStyle:
    """
    Einheitliche Typografie für Referenzgrafiken.
    """
    title_fontsize: int = 14
    subtitle_fontsize: int = 12
    label_fontsize: int = 12
    tick_fontsize: int = 11
    info_fontsize: int = 12

# ====== Eigenschaften des Modells ================

def expectation(model: BinomialModel) -> float:
    """
    Erwartungswert E(X) der Binomialverteilung.
    """
    return model.n * model.p


def variance(model: BinomialModel) -> float:
    """
    Varianz Var(X) der Binomialverteilung.
    """
    return model.n * model.p * (1 - model.p)


def sigma(model: BinomialModel) -> float:
    """
    Standardabweichung der Binomialverteilung.
    """
    return sqrt(variance(model))


# ======= 2. Wahrscheinlichkeiten =======


def prob_eq(model: BinomialModel, k: int) -> float:
    """
    Punktwahrscheinlichkeit P(X = k).
    """
    return binom.pmf(k, model.n, model.p)


def prob_le(model: BinomialModel, k: int) -> float:
    """
    Kumulative Wahrscheinlichkeit P(X <= k).
    """
    return binom.cdf(k, model.n, model.p)


def prob_ge(model: BinomialModel, k: int) -> float:
    """
    Kumulative Wahrscheinlichkeit P(X >= k).
    Diskrete Verteilung: >= k ist das Komplement von <= k-1
    """
    return 1.0 - binom.cdf(k - 1, model.n, model.p)


def prob_range(
    model: BinomialModel,
    k_min: int,
    k_max: int,
) -> float:
    """
    Bereichswahrscheinlichkeit P(k_min <= X <= k_max).
    """
    if k_min > k_max:
        raise ValueError("k_min must be <= k_max")

    return (
        binom.cdf(k_max, model.n, model.p)
        - binom.cdf(k_min - 1, model.n, model.p)
    )

# ======= 3. Grafische Darstellungen =======

def plot_binomial_distribution(
    model: BinomialModel,
    style: DistributionPlotStyle,
    *,
    typo=TypographyStyle,
    k_highlight: int | None = None,
    k_range: tuple[int, int] | None = None,
    save: str | None = None,
):
    """
    Grafische Darstellung der Binomialverteilung P(X = k).

    Optional:
    - Hervorhebung eines einzelnen k
    - Hervorhebung eines Bereichs [k_min, k_max]

    Keine Testentscheidung.
    Keine Interpretation.
    """

    n = model.n
    p = model.p

    # ----------------------------
    # relevanter Darstellungsbereich
    # ----------------------------
    mu = expectation(model)
    s = sigma(model)

    k_min = max(0, int(mu - style.sigma_range * s))
    k_max = min(n, int(mu + style.sigma_range * s))

    ks = np.arange(k_min, k_max + 1)
    probs = [prob_eq(model, k) for k in ks]

    fig, ax = plt.subplots(figsize=style.figsize)

    ax.bar(
        ks,
        probs,
        width=style.width,
        color=style.bar_color,
        edgecolor="black",
        linewidth=0.25,
    )

    # ----------------------------
    # optional: einzelnes Ereignis
    # ----------------------------
    if k_highlight is not None:
        ax.bar(
            k_highlight,
            prob_eq(model, k_highlight),
            color=style.highlight_color,
            width=style.width,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

    # ----------------------------
    # optional: Bereich
    # ----------------------------
    if k_range is not None:
        k_lo, k_hi = k_range
        mask = (ks >= k_lo) & (ks <= k_hi)
        ax.bar(
            ks[mask],
            [prob_eq(model, k) for k in ks[mask]],
            color=style.highlight_color,
            width=style.width,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

    # ----------------------------
    # Layout
    # ----------------------------
    ax.set_xlabel(r"$k$",fontsize=typo.label_fontsize)
    ax.set_ylabel(r"$P(X = k)$",fontsize=typo.label_fontsize)

    ax.tick_params(labelsize=typo.tick_fontsize)
    
    ax.text(
        0.5,
        1.05,
        rf"Binomialverteilung $X \sim \mathrm{{Bin}}({n},{p})$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.label_fontsize,
    )

    if style.grid:
        ax.grid(axis="y", alpha=0.5)

    ax.set_xlim(k_min - 0.5, k_max + 0.5)

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)


# =========== Binomial - mit Bereichen ===========

def plot_binomial_distribution_cdf(
    model: BinomialModel,
    style: DistributionPlotStyle,
    typo: TypographyStyle,
    *,
    k: int,
    direction: str = "le",   # "le" | "ge"
    save: str | None = None,
):
    """
    Grafische Darstellung kumulativer Wahrscheinlichkeiten
    der Binomialverteilung.

    direction = "le": P(X <= k)
    direction = "ge": P(X >= k)

    Die Darstellung zeigt die zugrunde liegenden Einzelwahrscheinlichkeiten
    als Balken und hebt den aufsummierten Bereich hervor.
    """

    if direction not in {"le", "ge"}:
        raise ValueError("direction must be 'le' or 'ge'")

    n = model.n
    p = model.p

    # --------------------------------------------------
    # relevanter Darstellungsbereich (wie zuvor)
    # --------------------------------------------------
    mu = expectation(model)
    s = sigma(model)

    k_min = max(0, int(mu - style.sigma_range * s))
    k_max = min(n, int(mu + style.sigma_range * s))

    ks = np.arange(k_min, k_max + 1)
    probs = [prob_eq(model, ki) for ki in ks]

    # --------------------------------------------------
    # kumulative Wahrscheinlichkeit
    # --------------------------------------------------
    if direction == "le":
        value = prob_le(model, k)
        mask = ks <= k
        title = r"Kumulative Wahrscheinlichkeit $P(X \leq k)$"
    else:
        value = prob_ge(model, k)
        mask = ks >= k
        title = r"Kumulative Wahrscheinlichkeit $P(X \geq k)$"

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=style.figsize)

    # Grundverteilung (alle Balken)
    ax.bar(
        ks,
        probs,
        width=style.width,
        color=style.bar_color,
        edgecolor="black",
        linewidth=0.5,
        zorder=1,
    )

    # Hervorgehobener Bereich (aufsummierte Ereignisse)
    ax.bar(
        ks[mask],
        [prob_eq(model, ki) for ki in ks[mask]],
        width=style.width,
        color=style.highlight_color,
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    ax.set_xlabel(r"$k$", fontsize=typo.label_fontsize)
    ax.set_ylabel(r"$P(X = k)$", fontsize=typo.label_fontsize)
    ax.tick_params(labelsize=typo.tick_fontsize)

    ax.set_xlim(k_min - 0.5, k_max + 0.5)

    if style.grid:
        ax.grid(axis="y", alpha=0.5)

    # --------------------------------------------------
    # Titel & Untertitel
    # --------------------------------------------------
    ax.text(
        0.5,
        1.09,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.title_fontsize,
    )

    ax.text(
        0.5,
        1.01,
        rf"Modell: $X \sim \mathrm{{Bin}}({n},{p})$ | Grenze: $k={k}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.subtitle_fontsize,
    )

    # --------------------------------------------------
    # Ergebniszeile
    # --------------------------------------------------
    if direction == "le":
        formula = r"$P(X \leq k)$"
    else:
        formula = r"$P(X \geq k)$"
    
    ax.text(
        0.5,
        -0.18,
        rf"{formula} = {value:.4f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=typo.info_fontsize,
    )

    # --------------------------------------------------
    # Speichern / Anzeigen
    # --------------------------------------------------
    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    return value


# =========== Verteilungsfunktion ===============
def plot_binomial_cdf(
    model: BinomialModel,
    style: DistributionPlotStyle,
    typo: TypographyStyle,
    *,
    k_mark: int | None = None,
    direction: str = "le",   # "le" | "ge"
    save: str | None = None,
):
    """
    Grafische Darstellung der Verteilungsfunktion
    F(k) = P(X <= k) der Binomialverteilung.

    Die Verteilungsfunktion wird als stufige, rechtsstetige Funktion dargestellt.
    """
    if direction not in {"le", "ge"}:
        raise ValueError("direction must be 'le' or 'ge'")
    
    n = model.n
    p = model.p

    # --------------------------------------------------
    # relevanter Darstellungsbereich
    # --------------------------------------------------
    mu = expectation(model)
    s = sigma(model)

    k_min = max(0, int(mu - style.sigma_range * s))
    k_max = min(n, int(mu + style.sigma_range * s))

    ks = np.arange(k_min, k_max + 1)
    Fs = [prob_le(model, k) for k in ks]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=style.figsize)

    ax.step(ks,Fs,where="post",color=style.bar_color,linewidth=2.0)

    # --------------------------------------------------
    # optionale Markierung eines Wertes
    # --------------------------------------------------
    Fk_le= prob_le(model, k_mark)
    if k_mark is not None:
        if direction == "le":
            Fk = Fk_le
            ax.vlines(k_mark,0,Fk,linestyle=":",color="gray",linewidth=1.5)
            ax.hlines(Fk,k_min,k_mark,linestyle=":",color="gray",linewidth=1.5)
            ax.vlines(k_min,0,Fk, linestyle="-", color="gray", linewidth=7,alpha=0.6)
            info = rf"$P(X \leq {k_mark}) = {Fk:.4f}$"
        
        else:
            Fk = prob_ge(model, k_mark)
    
            ax.vlines(k_mark, 0, Fk_le, linestyle=":", color="gray", linewidth=1.5)
            ax.hlines(Fk_le, k_mark, k_max, linestyle=":", color="gray", linewidth=1.5)
            ax.vlines(k_max, Fk_le,1, linestyle="-", color="gray", linewidth=7,alpha=0.6)

            info = rf"$P(X \geq {k_mark}) = {Fk:.4f}$"
       
        ax.plot(k_mark,Fk_le,"o",color=style.highlight_color,zorder=3)

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    ax.set_xlim(k_min, k_max)
    ax.set_ylim(0, 1.02)

    ax.set_xlabel(r"$k$", fontsize=typo.label_fontsize)
    ax.set_ylabel(r"$F(k) = P(X \leq k)$", fontsize=typo.label_fontsize)
    ax.tick_params(labelsize=typo.tick_fontsize)

    if style.grid:
        ax.grid(alpha=0.6)

    # --------------------------------------------------
    # Titel
    # --------------------------------------------------
    ax.text(
        0.5,
        1.09,
        "Verteilungsfunktion der Binomialverteilung",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.title_fontsize,
    )

    ax.text(
        0.5,
        1.01,
        rf"$X \sim \mathrm{{Bin}}({n},{p})$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.subtitle_fontsize,
    )

    # --------------------------------------------------
    # optionale Ergebniszeile
    # --------------------------------------------------
    if k_mark is not None:
        ax.text(
            0.5,
            -0.18,
            info,
            #rf"$F({k_mark}) = P(X \leq {k_mark}) = {Fk:.4f}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=typo.info_fontsize,
        )

    # --------------------------------------------------
    # Speichern / Anzeigen
    # --------------------------------------------------
    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_prediction_interval_absolute_distribution(
    model: BinomialModel,
    gamma: float,
    style: DistributionPlotStyle,
    typo: TypographyStyle,
    *,
    save: str | None = None,
):
    """
    Absolutes Prognoseintervall im Stichprobenraum.

    Darstellung der Binomialverteilung P(X = k).
    Werte außerhalb des Prognoseintervalls werden rot markiert.
    Kein Test. Keine Entscheidung.
    """

    n, p = model.n, model.p
    z = sqrt(2) * binom._ppf((1 + gamma) / 2, 1, 0.5)  # nur symbolisch stabil
    mu = expectation(model)
    s = sigma(model)

    k_L = mu - z * s
    k_R = mu + z * s

    k_L_int = int(np.ceil(k_L))
    k_R_int = int(np.floor(k_R))

    # Darstellungsbereich
    k_min = max(0, int(mu - style.sigma_range * s))
    k_max = min(n, int(mu + style.sigma_range * s))

    ks = np.arange(k_min, k_max + 1)
    probs = [prob_eq(model, k) for k in ks]

    fig, ax = plt.subplots(figsize=style.figsize)

    # Grundverteilung
    ax.bar(
        ks,
        probs,
        width=style.width,
        color=style.bar_color,
        edgecolor="black",
        linewidth=0.3,
        zorder=1,
    )

    # Tails (außerhalb des PI)
    mask = (ks < k_L_int) | (ks > k_R_int)
    ax.bar(
        ks[mask],
        [prob_eq(model, k) for k in ks[mask]],
        width=style.width,
        color="tab:red",
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )

    # Layout
    ax.set_xlabel(r"$k$", fontsize=typo.label_fontsize)
    ax.set_ylabel(r"$P(X = k)$", fontsize=typo.label_fontsize)
    ax.tick_params(labelsize=typo.tick_fontsize)

    ax.text(
        0.5,
        1.08,
        "Absolutes Prognoseintervall (Stichprobenraum)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.title_fontsize,
    )

    ax.text(
        0.5,
        1.01,
        rf"$X \sim \mathrm{{Bin}}({n},{p}),\; \gamma={gamma}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.subtitle_fontsize,
    )

    ax.text(
        0.5,
        -0.18,
        rf"{gamma*100:.0f}%-PI: "
        rf"[{k_L:.2f}; {k_R:.2f}] | äquivalent: [{k_L_int}; {k_R_int}]",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=typo.info_fontsize,
    )

    if style.grid:
        ax.grid(axis="y", alpha=0.5)

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_prediction_interval_absolute_cdf(
    model: BinomialModel,
    gamma: float,
    style: DistributionPlotStyle,
    typo: TypographyStyle,
    *,
    save: str | None = None,
):
    """
    Verteilungsfunktion mit markiertem absolutem Prognoseintervall.
    Zeigt die beiden Tails (1 - gamma) im CDF-Raum.
    """

    n, p = model.n, model.p
    z = sqrt(2) * binom._ppf((1 + gamma) / 2, 1, 0.5)
    mu = expectation(model)
    s = sigma(model)

    k_L = mu - z * s
    k_R = mu + z * s

    k_L_int = int(np.ceil(k_L))
    k_R_int = int(np.floor(k_R))

    k_min = max(0, int(mu - style.sigma_range * s))
    k_max = min(n, int(mu + style.sigma_range * s))

    ks = np.arange(k_min, k_max + 1)
    Fs = [prob_le(model, k) for k in ks]

    fig, ax = plt.subplots(figsize=style.figsize)

    ax.step(ks, Fs, where="post", color=style.bar_color, linewidth=2.0)

    # Linke Tail
    F_L = prob_le(model, k_L_int - 1)
    ax.vlines(k_min, 0, F_L, color="tab:red", linewidth=6, alpha=0.6)

    # Rechte Tail
    F_R = prob_le(model, k_R_int)
    ax.vlines(k_max, F_R, 1, color="tab:red", linewidth=6, alpha=0.6)

    ax.set_xlim(k_min, k_max)
    ax.set_ylim(0, 1.02)

    ax.set_xlabel(r"$k$", fontsize=typo.label_fontsize)
    ax.set_ylabel(r"$F(k) = P(X \leq k)$", fontsize=typo.label_fontsize)
    ax.tick_params(labelsize=typo.tick_fontsize)

    ax.text(
        0.5,
        1.08,
        "Verteilungsfunktion zum absoluten Prognoseintervall",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.title_fontsize,
    )

    ax.text(
        0.5,
        1.01,
        rf"$X \sim \mathrm{{Bin}}({n},{p}),\; \gamma={gamma}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.subtitle_fontsize,
    )

    ax.text(
        0.5,
        -0.18,
        rf"Außerhalb des PI: $P(X < {k_L_int}) + P(X > {k_R_int}) \approx {1-gamma:.3f}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=typo.info_fontsize,
    )

    if style.grid:
        ax.grid(alpha=0.6)

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)

# =========== 4. Tests  ==========================
# === Wann wird aus Wahrscheinlichkeit eine Entscheidung? ===

def rejection_region_left(test: BinomialTest) -> list[int]:
    """
    Linksseitiger Ablehnungsbereich für H0: p = p0.
    """
    model = BinomialModel(test.n, test.p0)

    cumulative = 0.0
    region = []

    for k in range(test.n + 1):
        cumulative += prob_eq(model, k)
        if cumulative <= test.alpha:
            region.append(k)
        else:
            break

    return region

def rejection_region_right(test: BinomialTest) -> list[int]:
    """
    Rechtsseitiger Ablehnungsbereich für H0: p = p0.
    """
    model = BinomialModel(test.n, test.p0)

    cumulative = 0.0
    region = []

    for k in reversed(range(test.n + 1)):
        cumulative += prob_eq(model, k)
        if cumulative <= test.alpha:
            region.append(k)
        else:
            break

    return sorted(region)


def rejection_region_two_sided(test: BinomialTest) -> tuple[list[int], list[int]]:
    """
    Zweiseitiger Ablehnungsbereich (links und rechts).
    """
    half_alpha = test.alpha / 2

    left_test = BinomialTest(test.n, test.p0, half_alpha)
    right_test = BinomialTest(test.n, test.p0, half_alpha)

    left = rejection_region_left(left_test)
    right = rejection_region_right(right_test)

    return left, right


# ================  5. Simulationen  =======================

def simulate_test(
    sim: BinomialSimulation,
    rejection_region: set[int],
) -> float:
    """
    Simuliert die Ablehnungswahrscheinlichkeit eines Tests.

    Gibt die empirische Ablehnungsrate zurück.
    """
    rng = np.random.default_rng(sim.seed)
    rejections = 0

    for _ in range(sim.m):
        x = rng.binomial(sim.n, sim.p_true)
        if x in rejection_region:
            rejections += 1

    return rejections / sim.m


# ======= 6. Zusammengesetzte Referenzgrafiken =======

def plot_binomial_model_with_rejection_region(
    model: BinomialModel,
    test: BinomialTest,
    style: DistributionPlotStyle,
    typo: TypographyStyle,
    *,
    k_obs: int | None = None,
    save: str | None = None,
):
    """
    Referenzgrafik:
    Binomialverteilung mit zweiseitigem Ablehnungsbereich.

    - Ablehnungsbereich = Setzung
    - Beobachtung k_obs optional
    - keine Testentscheidung
    """

    n = model.n
    p = model.p

    # --------------------------------------------------
    # Ablehnungsbereich (zweiseitig, equal tails)
    # --------------------------------------------------
    left, right = rejection_region_two_sided(test)
    reject_set = set(left) | set(right)

    # --------------------------------------------------
    # relevanter Darstellungsbereich
    # --------------------------------------------------
    mu = expectation(model)
    s = sigma(model)

    k_min = max(0, int(mu - style.sigma_range * s))
    k_max = min(n, int(mu + style.sigma_range * s))

    ks = np.arange(k_min, k_max + 1)
    probs = [prob_eq(model, k) for k in ks]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=style.figsize)

    # Grundverteilung
    ax.bar(
        ks,
        probs,
        width=style.width,
        color=style.bar_color,
        edgecolor="black",
        linewidth=0.3,
        zorder=1,
    )

    # Ablehnungsbereich
    ks_reject = [k for k in ks if k in reject_set]
    ax.bar(
        ks_reject,
        [prob_eq(model, k) for k in ks_reject],
        width=style.width,
        color=style.highlight_color,
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
        label=rf"Ablehnungsbereich ($\alpha={test.alpha}$)",
    )

    # optionale Beobachtung
    if k_obs is not None:
        ax.bar(
            k_obs,
            prob_eq(model, k_obs),
            width=style.width,
            color="black",
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    ax.set_xlim(k_min - 0.5, k_max + 0.5)

    ax.set_xlabel(r"$k$", fontsize=typo.label_fontsize)
    ax.set_ylabel(r"$P(X = k)$", fontsize=typo.label_fontsize)
    ax.tick_params(labelsize=typo.tick_fontsize)

    if style.grid:
        ax.grid(axis="y", alpha=0.5)

    # --------------------------------------------------
    # Titel
    # --------------------------------------------------
    ax.text(
        0.5,
        1.08,
        "Binomialverteilung mit Ablehnungsbereich",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.title_fontsize,
    )

    ax.text(
        0.5,
        1.01,
        rf"$X \sim \mathrm{{Bin}}({n},{p})$  |  $H_0: p={test.p0}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.subtitle_fontsize,
    )

    # --------------------------------------------------
    # optionale Information (semantisch, nicht grafisch)
    # --------------------------------------------------
    if k_obs is not None:
        p_val = (
            prob_le(model, k_obs)
            if k_obs in left
            else prob_ge(model, k_obs)
        )

        ax.text(
            0.5,
            -0.18,
            rf"$k_{{obs}}={k_obs}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=typo.info_fontsize,
        )

    # --------------------------------------------------
    # Ausgabe
    # --------------------------------------------------
    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)


# ---------- absolures Prognoseintervall -------------

def plot_prediction_interval_absolute(
    model: BinomialModel,
    gamma: float,
    style: DistributionPlotStyle,
    typo: TypographyStyle,
    *,
    save: str | None = None,
):
    """
    Grafische Darstellung eines absoluten Prognoseintervalls
    für X ~ Bin(n,p).

    - Balken innerhalb des Prognoseintervalls: neutral
    - Balken außerhalb: rot
    - keine Testentscheidung
    """

    n = model.n
    p = model.p

    # --------------------------------------------------
    # Prognoseintervall (Normalapproximation)
    # --------------------------------------------------
    z = sqrt(2) * binom.ppf((1 + gamma) / 2, 1, 0.5) / sqrt(1)  # nur zur Klarheit
    mu = expectation(model)
    s = sigma(model)

    k_L = mu - z * s
    k_R = mu + z * s

    k_L_int = int(np.ceil(k_L))
    k_R_int = int(np.floor(k_R))

    # --------------------------------------------------
    # Darstellungsbereich
    # --------------------------------------------------
    k_min = max(0, int(mu - style.sigma_range * s))
    k_max = min(n, int(mu + style.sigma_range * s))

    ks = np.arange(k_min, k_max + 1)
    probs = [prob_eq(model, k) for k in ks]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=style.figsize)

    # neutrale Balken
    ax.bar(
        ks,
        probs,
        width=style.width,
        color=style.bar_color,
        edgecolor="black",
        linewidth=0.3,
        zorder=1,
    )

    # Tails (außerhalb des PI)
    mask = (ks < k_L_int) | (ks > k_R_int)
    ax.bar(
        ks[mask],
        [prob_eq(model, k) for k in ks[mask]],
        width=style.width,
        color="tab:red",
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    ax.set_xlim(k_min - 0.5, k_max + 0.5)
    ax.set_xlabel(r"$k$", fontsize=typo.label_fontsize)
    ax.set_ylabel(r"$P(X = k)$", fontsize=typo.label_fontsize)
    ax.tick_params(labelsize=typo.tick_fontsize)

    if style.grid:
        ax.grid(axis="y", alpha=0.5)

    # --------------------------------------------------
    # Titel
    # --------------------------------------------------
    ax.text(
        0.5,
        1.08,
        "Absolutes Prognoseintervall",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.title_fontsize,
    )

    ax.text(
        0.5,
        1.01,
        rf"$X \sim \mathrm{{Bin}}({n},{p}),\; \gamma={gamma}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.subtitle_fontsize,
    )

    # --------------------------------------------------
    # Ergebniszeile
    # --------------------------------------------------
    ax.text(
        0.5,
        -0.18,
        rf"{gamma*100:.0f}%-PI: [{k_L:.2f}; {k_R:.2f}]  |  äquivalent: [{k_L_int}; {k_R_int}]",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=typo.info_fontsize,
    )

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)

# ---------- relatives Prognoseintervall ---------

def plot_prediction_interval_absolute(
    model: BinomialModel,
    gamma: float,
    style: DistributionPlotStyle,
    typo: TypographyStyle,
    *,
    save: str | None = None,
):
    """
    Grafische Darstellung eines absoluten Prognoseintervalls
    für X ~ Bin(n,p).

    - Balken innerhalb des Prognoseintervalls: neutral
    - Balken außerhalb: rot
    - keine Testentscheidung
    """

    n = model.n
    p = model.p

    # --------------------------------------------------
    # Prognoseintervall (Normalapproximation)
    # --------------------------------------------------
    z = sqrt(2) * binom.ppf((1 + gamma) / 2, 1, 0.5) / sqrt(1)  # nur zur Klarheit
    mu = expectation(model)
    s = sigma(model)

    k_L = mu - z * s
    k_R = mu + z * s

    k_L_int = int(np.ceil(k_L))
    k_R_int = int(np.floor(k_R))

    # --------------------------------------------------
    # Darstellungsbereich
    # --------------------------------------------------
    k_min = max(0, int(mu - style.sigma_range * s))
    k_max = min(n, int(mu + style.sigma_range * s))

    ks = np.arange(k_min, k_max + 1)
    probs = [prob_eq(model, k) for k in ks]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=style.figsize)

    # neutrale Balken
    ax.bar(
        ks,
        probs,
        width=style.width,
        color=style.bar_color,
        edgecolor="black",
        linewidth=0.3,
        zorder=1,
    )

    # Tails (außerhalb des PI)
    mask = (ks < k_L_int) | (ks > k_R_int)
    ax.bar(
        ks[mask],
        [prob_eq(model, k) for k in ks[mask]],
        width=style.width,
        color="tab:red",
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    ax.set_xlim(k_min - 0.5, k_max + 0.5)
    ax.set_xlabel(r"$k$", fontsize=typo.label_fontsize)
    ax.set_ylabel(r"$P(X = k)$", fontsize=typo.label_fontsize)
    ax.tick_params(labelsize=typo.tick_fontsize)

    if style.grid:
        ax.grid(axis="y", alpha=0.5)

    # --------------------------------------------------
    # Titel
    # --------------------------------------------------
    ax.text(
        0.5,
        1.08,
        "Absolutes Prognoseintervall",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.title_fontsize,
    )

    ax.text(
        0.5,
        1.01,
        rf"$X \sim \mathrm{{Bin}}({n},{p}),\; \gamma={gamma}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.subtitle_fontsize,
    )

    # --------------------------------------------------
    # Ergebniszeile
    # --------------------------------------------------
    ax.text(
        0.5,
        -0.18,
        rf"{gamma*100:.0f}%-PI: [{k_L:.2f}; {k_R:.2f}]  |  äquivalent: [{k_L_int}; {k_R_int}]",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=typo.info_fontsize,
    )

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)

# --------- absolutes PI ---------

def plot_prediction_interval_absolute_distribution(
    model: BinomialModel,
    gamma: float,
    style: DistributionPlotStyle,
    typo: TypographyStyle,
    *,
    save: str | None = None,
):
    """
    Absolutes Prognoseintervall im Stichprobenraum.

    - Verteilung X ~ Bin(n,p)
    - Prognoseintervall [np - z*sqrt(np(1-p)), np + z*sqrt(np(1-p))]
    - Balken außerhalb des Intervalls rot (beide Tails)
    - kein Test, keine Entscheidung
    """

    n = model.n
    p = model.p

    # --------------------------------------------------
    # Prognoseintervall (stetig + diskret äquivalent)
    # --------------------------------------------------
    z = NormalDist().inv_cdf((1 + gamma) / 2)
    mu = expectation(model)
    s = sigma(model)

    k_L = mu - z * s
    k_R = mu + z * s

    k_L_int = int(np.ceil(k_L))
    k_R_int = int(np.floor(k_R))

    # --------------------------------------------------
    # Darstellungsbereich
    # --------------------------------------------------
    k_min = max(0, int(mu - style.sigma_range * s))
    k_max = min(n, int(mu + style.sigma_range * s))

    ks = np.arange(k_min, k_max + 1)
    probs = [prob_eq(model, k) for k in ks]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=style.figsize)

    # neutrale Balken
    ax.bar(
        ks,
        probs,
        width=style.width,
        color=style.bar_color,
        edgecolor="black",
        linewidth=0.3,
        zorder=1,
    )

    # Tails (außerhalb des Prognoseintervalls)
    mask = (ks < k_L_int) | (ks > k_R_int)
    ax.bar(
        ks[mask],
        [prob_eq(model, k) for k in ks[mask]],
        width=style.width,
        color="tab:red",
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    ax.set_xlim(k_min - 0.5, k_max + 0.5)
    ax.set_xlabel(r"$k$", fontsize=typo.label_fontsize)
    ax.set_ylabel(r"$P(X = k)$", fontsize=typo.label_fontsize)
    ax.tick_params(labelsize=typo.tick_fontsize)

    if style.grid:
        ax.grid(axis="y", alpha=0.5)

    # --------------------------------------------------
    # Titel
    # --------------------------------------------------
    ax.text(
        0.5,
        1.08,
        "Absolutes Prognoseintervall (Stichprobenraum)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.title_fontsize,
    )

    ax.text(
        0.5,
        1.01,
        rf"$X \sim \mathrm{{Bin}}({n},{p}),\; \gamma={gamma}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.subtitle_fontsize,
    )

    # --------------------------------------------------
    # Ergebniszeile
    # --------------------------------------------------
    ax.text(
        0.5,
        -0.18,
        rf"{gamma*100:.0f}%-PI: [{k_L:.2f}; {k_R:.2f}]  |  äquivalent: [{k_L_int}; {k_R_int}]",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=typo.info_fontsize,
    )

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)

# -------- abs. PI - Verteilungsfunktion ------

def plot_prediction_interval_absolute_cdf(
    model: BinomialModel,
    gamma: float,
    style: DistributionPlotStyle,
    typo: TypographyStyle,
    *,
    save: str | None = None,
):
    """
    Verteilungsfunktion der Binomialverteilung
    mit Markierung der beiden Randwahrscheinlichkeiten
    zum absoluten Prognoseintervall.
    """

    n = model.n
    p = model.p

    # --------------------------------------------------
    # Prognoseintervall
    # --------------------------------------------------
    z = NormalDist().inv_cdf((1 + gamma) / 2)
    mu = expectation(model)
    s = sigma(model)

    k_L = mu - z * s
    k_R = mu + z * s

    k_L_int = int(np.ceil(k_L))
    k_R_int = int(np.floor(k_R))

    # --------------------------------------------------
    # Darstellungsbereich
    # --------------------------------------------------
    k_min = max(0, int(mu - style.sigma_range * s))
    k_max = min(n, int(mu + style.sigma_range * s))

    ks = np.arange(k_min, k_max + 1)
    Fs = [prob_le(model, k) for k in ks]

    # --------------------------------------------------
    # Wahrscheinlichkeiten der Tails
    # --------------------------------------------------
    p_left = prob_le(model, k_L_int - 1)
    p_right = prob_ge(model, k_R_int + 1)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=style.figsize)

    ax.step(
        ks,
        Fs,
        where="post",
        color=style.bar_color,
        linewidth=2.0,
        zorder=1,
    )

    # linke Tail-Markierung
    ax.vlines(
        k_L_int,
        0,
        p_left,
        linestyle=":",
        color="tab:red",
        linewidth=1.5,
    )
    ax.hlines(
        p_left,
        k_min,
        k_L_int,
        linestyle="-",
        color="tab:red",
        linewidth=6,
        alpha=0.6,
    )

    # rechte Tail-Markierung
    ax.vlines(
        k_R_int,
        p_left,
        1,
        linestyle=":",
        color="tab:red",
        linewidth=1.5,
    )
    ax.hlines(
        p_left,
        k_R_int,
        k_max,
        linestyle="-",
        color="tab:red",
        linewidth=6,
        alpha=0.6,
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    ax.set_xlim(k_min, k_max)
    ax.set_ylim(0, 1.02)

    ax.set_xlabel(r"$k$", fontsize=typo.label_fontsize)
    ax.set_ylabel(r"$F(k) = P(X \leq k)$", fontsize=typo.label_fontsize)
    ax.tick_params(labelsize=typo.tick_fontsize)

    if style.grid:
        ax.grid(alpha=0.6)

    # --------------------------------------------------
    # Titel
    # --------------------------------------------------
    ax.text(
        0.5,
        1.08,
        "Verteilungsfunktion und Prognoseintervall",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.title_fontsize,
    )

    ax.text(
        0.5,
        1.01,
        rf"$X \sim \mathrm{{Bin}}({n},{p}),\; \gamma={gamma}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=typo.subtitle_fontsize,
    )

    # --------------------------------------------------
    # Ergebniszeile
    # --------------------------------------------------
    ax.text(
        0.5,
        -0.18,
        rf"Außerhalb des {gamma*100:.0f}%-PI: "
        rf"$P(X < {k_L_int}) + P(X > {k_R_int}) \approx {1-gamma:.3f}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=typo.info_fontsize,
    )

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    