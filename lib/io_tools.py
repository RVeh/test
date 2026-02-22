from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from datetime import datetime
import json
import re
from typing import Any, Mapping


# ---------- Basisordner (Repo-Root robust bestimmen) ----------

def repo_root(start: Path | None = None) -> Path:
    """
    Findet den Projekt-Root robust:
    - Startpunkt: aktuelles Arbeitsverzeichnis (Binder/Notebook) oder gegeben
    - sucht nach typischen Markern (README.md, requirements.txt, notebooks/)
    """
    p = (start or Path.cwd()).resolve()
    for _ in range(8):
        if (p / "README.md").exists() or (p / "requirements.txt").exists() or (p / "notebooks").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return (start or Path.cwd()).resolve()


def ensure_dir(*parts: str, root: Path | None = None) -> Path:
    d = (root or repo_root()) / Path(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------- Parameter → stabiler Tag ----------

def _normalize_value(v: Any) -> str:
    if isinstance(v, float):
        # 0.28 -> "0p28" (punktfrei, dateiname-sicher)
        s = f"{v:.6g}".replace(".", "p")
        return s
    return str(v)


def params_tag(params: Mapping[str, Any] | Any) -> str:
    """
    Baut einen kurzen, dateinamen-sicheren Parameter-Tag.

    Akzeptiert:
    - dict
    - dataclass
    """
    if is_dataclass(params):
        params = asdict(params)
    if not isinstance(params, Mapping):
        raise TypeError("params_tag erwartet dict oder dataclass")

    items = []
    for k in sorted(params.keys()):
        v = params[k]
        items.append(f"{k}-{_normalize_value(v)}")
    raw = "__".join(items)

    # Dateiname säubern
    raw = re.sub(r"[^A-Za-z0-9_\-p]+", "", raw)
    return raw[:160]  # nicht zu lang


# ---------- Output-Dateiname erzeugen ----------

def figure_path(
    activity: str,
    name: str,
    params: Mapping[str, Any] | Any,
    *,
    ext: str = "pdf",
    root: Path | None = None,
) -> Path:
    """
    Liefert einen reproduzierbaren Pfad: fig/<activity>/<name>__<params>.pdf
    """
    out = ensure_dir("fig", activity, root=root)
    tag = params_tag(params)
    return out / f"{name}__{tag}.{ext}"


def table_path(
    activity: str,
    name: str,
    params: Mapping[str, Any] | Any,
    *,
    ext: str = "csv",
    root: Path | None = None,
) -> Path:
    out = ensure_dir("tab", activity, root=root)
    tag = params_tag(params)
    return out / f"{name}__{tag}.{ext}"


def save_metadata(
    activity: str,
    name: str,
    params: Mapping[str, Any] | Any,
    *,
    extra: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    """
    Speichert Metadaten neben den Outputs (für Referenzierbarkeit).
    """
    meta_dir = ensure_dir("tab", activity, "_meta", root=root)
    payload = {"name": name, "params": asdict(params) if is_dataclass(params) else dict(params)}
    if extra:
        payload["extra"] = dict(extra)
    path = meta_dir / f"{name}__{params_tag(params)}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------- LaTeX Beamer Snippets ----------

def beamer_includegraphics(path: Path, *, width: str = r"0.9\linewidth") -> str:
    """
    Gibt eine fertige \\includegraphics-Zeile zurück (relativ zum Repo-Root).
    """
    root = repo_root()
    rel = path.resolve().relative_to(root)
    rel_posix = rel.as_posix()
    return rf"\includegraphics[width={width}]{{{rel_posix}}}"