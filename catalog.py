"""
Catálogo de resolución: la tupla parcial de la consulta se filtra contra las tuplas
completas de las series publicadas (data/catalogo/series.csv).

    from catalog import Catalog
    cat = Catalog.load()
    cat.resolve(indicador="imacec")                 # -> Resolucion(estado="resuelto" | "aclarar" | "no_existe")
    cat.espacio_trabajo().filter(frecuencia="trimestral").collapse()

Catorce dimensiones por serie, de tres clases:
  - lingüísticas (8): las enuncia el usuario; las produce el extractor (NER + normalización o cabeza)
  - rara (1): referencia; sólo si la consulta nombra una compilación ("base 2013")
  - estructurales (5): vigencia, empalme, info_historica, apertura, region_fija; nunca se mencionan,
    se fijan por defecto o se derivan del catálogo
Una dimensión ausente (None) no restringe. Una lista significa "cualquiera de".
Sin dependencias fuera de la biblioteca estándar, para poder usarse en inferencia.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PATH = Path(__file__).parent / "data" / "catalogo" / "series.csv"

DIMS_LINGUISTICAS = ["indicador", "frecuencia", "estacionalidad", "presentacion", "valoracion",
                     "actividad", "region", "gasto"]
DIMS_RARAS = ["referencia"]
DIMS_ESTRUCTURALES = ["vigencia", "empalme", "info_historica", "apertura", "region_fija"]
DIMS = DIMS_LINGUISTICAS + DIMS_RARAS + DIMS_ESTRUCTURALES

# Dimensiones de identidad de un cuadro (la tupla del paper); las de categoría identifican la serie.
DIMS_CATEGORIA = ["actividad", "region", "gasto"]
DIMS_CUADRO = [d for d in DIMS if d not in DIMS_CATEGORIA]

# Valores que existen en el catálogo pero no en el vocabulario del modelo (nadie los pregunta).
VALORES_MARGINALES: dict[str, set[Any]] = {
    "presentacion": {"tasa"},
    "estacionalidad": {"tendencia_ciclo"},
    "region": {"extrarregional"},
}

# Preferencias (defaults blandos) en orden de aplicación: se imponen sólo si el conjunto candidato
# contiene el valor preferido; si no, no restringen. Tabla institucional: mantener y versionar.
PREFERENCIAS: list[tuple[str, Any]] = [
    ("vigencia", "vigente"),
    ("info_historica", False),
    ("referencia", 2018),
    ("presentacion", "nivel"),
    ("valoracion", "encadenado"),
    ("empalme", True),          # último: desempate entre cuadros por lo demás equivalentes
]

_BOOL = {"True": True, "False": False}


def _parse(row: dict[str, str]) -> dict[str, Any]:
    r: dict[str, Any] = dict(row)
    for k in ("empalme", "info_historica"):
        r[k] = _BOOL[r[k]]
    r["referencia"] = int(r["referencia"])
    for k in ("actividad", "region", "gasto", "region_fija", "categoria"):
        r[k] = r[k] or None
    return r


@dataclass
class Resolucion:
    estado: str                                   # "resuelto" | "aclarar" | "no_existe"
    consulta: dict[str, Any]                      # tupla parcial recibida
    candidatos: "Catalog"
    colapsadas: dict[str, Any] = field(default_factory=dict)   # dims fijadas por propagación
    supuestos: dict[str, Any] = field(default_factory=dict)    # dims fijadas por preferencia
    aclarar: tuple[str, Counter] | None = None                 # eje y opciones de la pregunta aclaratoria

    @property
    def cuadros(self) -> dict[str, str]:
        return self.candidatos.cuadros()

    @property
    def series(self) -> list[str]:
        return [s["serie_id"] for s in self.candidatos]


class Catalog:
    def __init__(self, series: list[dict[str, Any]]):
        self.series = series

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "Catalog":
        with open(path, encoding="utf-8", newline="") as f:
            return cls([_parse(r) for r in csv.DictReader(f)])

    def __len__(self) -> int:
        return len(self.series)

    def __iter__(self):
        return iter(self.series)

    # ---- filtrado ------------------------------------------------------------
    def filter(self, **q: Any) -> "Catalog":
        """Conserva las series compatibles con la tupla parcial q (None = no restringe)."""
        conds = {k: v for k, v in q.items() if v is not None}
        unknown = set(conds) - set(DIMS)
        if unknown:
            raise KeyError(f"dimensiones desconocidas: {sorted(unknown)}; válidas: {DIMS}")

        def ok(s: dict[str, Any]) -> bool:
            for k, v in conds.items():
                if isinstance(v, (list, tuple, set, frozenset)):
                    if s[k] not in v:
                        return False
                elif s[k] != v:
                    return False
            return True

        return Catalog([s for s in self.series if ok(s)])

    def prefer(self, dim: str, value: Any) -> "Catalog":
        """Default blando: restringe a `value` sólo si algún candidato lo tiene."""
        sub = self.filter(**{dim: value})
        return sub if len(sub) else self

    def sin_marginales(self) -> "Catalog":
        return Catalog([s for s in self.series
                        if not any(s[d] in vals for d, vals in VALORES_MARGINALES.items())])

    def espacio_trabajo(self) -> "Catalog":
        """Vista por defecto: publicaciones vigentes, sin valores marginales."""
        return self.filter(vigencia="vigente").sin_marginales()

    # ---- inspección ----------------------------------------------------------
    def values(self, dim: str) -> Counter:
        return Counter(s[dim] for s in self.series)

    def collapse(self, dims: Iterable[str] = DIMS_CUADRO) -> dict[str, Any]:
        """Dimensiones que toman un único valor en el conjunto candidato (propagación de restricciones)."""
        out = {}
        for d in dims:
            vals = {s[d] for s in self.series}
            if len(vals) == 1:
                out[d] = vals.pop()
        return out

    def cuadros(self) -> dict[str, str]:
        return {s["cuadro_id"]: s["cuadro"] for s in self.series}

    def widest_dim(self, dims: Iterable[str] = DIMS_CUADRO) -> tuple[str, Counter] | None:
        """Dimensión de mayor cardinalidad en el residual: eje para la pregunta aclaratoria."""
        best = None
        for d in dims:
            c = self.values(d)
            if len(c) > 1 and (best is None or len(c) > len(best[1])):
                best = (d, c)
        return best

    # ---- resolución ----------------------------------------------------------
    def resolve(self, **q: Any) -> Resolucion:
        """
        Filtra la tupla parcial q, propaga restricciones y aplica preferencias sobre lo que queda
        indeterminado. Resuelve cuando queda un solo cuadro; si no, señala el eje a aclarar.
        Si q nombra una referencia se abre el universo histórico; si no, se trabaja sobre vigentes.
        """
        q = {k: v for k, v in q.items() if v is not None}
        base = self.sin_marginales() if q.get("referencia") not in (None, 2018) else self.espacio_trabajo()
        cands = base.filter(**q)
        if not len(cands):
            return Resolucion("no_existe", q, cands)

        supuestos: dict[str, Any] = {}
        for dim, val in PREFERENCIAS:
            if dim in q or len(cands.values(dim)) == 1:
                continue
            nxt = cands.prefer(dim, val)
            if len(nxt) < len(cands):
                supuestos[dim] = val
                cands = nxt

        colapsadas = {d: v for d, v in cands.collapse().items() if d not in q and d not in supuestos}
        if len(cands.cuadros()) == 1:
            return Resolucion("resuelto", q, cands, colapsadas, supuestos)
        return Resolucion("aclarar", q, cands, colapsadas, supuestos, aclarar=cands.widest_dim())
