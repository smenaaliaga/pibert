"""
Catálogo de resolución: la tupla parcial de la consulta se filtra contra las tuplas
completas de las series publicadas (data/catalogo/series.csv).

    from catalog import Catalog
    cat = Catalog.load()
    cat.resolve(indicador="imacec")                 # -> Resolucion(estado="resuelto" | "aclarar" | "no_existe")
    cat.resolve_consulta({"activ_cls": 1}, indicador="imacec", estacionalidad="original")   # Cuadro 2 del paper
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

# Eje de desagregación (paper, sección 4.2): dimensión de categoría -> cabeza clasificadora
CABEZAS_EJE = {"actividad": "activ_cls", "region": "reg_cls", "gasto": "gast_cls"}
# Dimensiones en que difieren los cortes alternativos de un mismo dato (o los cuadros de cada región)
CORTES = {"apertura", "region_fija"}
_AGREGADO = (None, "total")

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
    modo: str | None = None                       # resolve_consulta: "agregada" | "categorias" | "cuadro"
    destacadas: dict[str, list[str]] = field(default_factory=dict)  # categorías mencionadas en modo "cuadro"

    @property
    def cuadros(self) -> dict[str, str]:
        return self.candidatos.cuadros()

    @property
    def series(self) -> list[str]:
        """Series distintas; una misma serie puede figurar en más de un cuadro (pares cuadro-serie)."""
        return list(dict.fromkeys(s["serie_id"] for s in self.candidatos))


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
    def _base(self, q: dict[str, Any]) -> "Catalog":
        """Si q nombra una referencia se abre el universo histórico; si no, se trabaja sobre vigentes."""
        return self.sin_marginales() if q.get("referencia") not in (None, 2018) else self.espacio_trabajo()

    def resolve(self, **q: Any) -> Resolucion:
        """
        Filtra la tupla parcial q, propaga restricciones y aplica preferencias sobre lo que queda
        indeterminado. Resuelve cuando queda un solo cuadro; si no, señala el eje a aclarar.
        """
        q = {k: v for k, v in q.items() if v is not None}
        return self._resolver(self._base(q).filter(**q), q)

    def resolve_consulta(self, cabezas: dict[str, Any] | None = None, **q: Any) -> Resolucion:
        """
        Resolución de la salida del extractor con el eje de desagregación (paper, sección 4.2).
        Un eje (actividad, región, gasto) está activo si hay un span de ese tipo o su cabeza vale 1,
        y la combinación fija la cardinalidad de la respuesta (Cuadro 2):

            span  cabeza
             -      0     fila agregada del cuadro                              modo "agregada"
             sí     0     las categorías mencionadas                            modo "categorias"
             -      1     el cuadro en su totalidad                             modo "cuadro"
             sí     1     el cuadro en su totalidad, con las mencionadas        modo "cuadro" + destacadas

        `cabezas` trae activ_cls/reg_cls/gast_cls; q, las dimensiones normalizadas, donde
        actividad/region/gasto admiten una lista de categorías. Los cortes alternativos de un
        mismo dato (cuadros que difieren sólo en apertura o región fija) no piden aclaración.
        """
        cabezas = cabezas or {}
        activa = {e: int(cabezas.get(h, 0)) == 1 for e, h in CABEZAS_EJE.items()}
        q = {k: v for k, v in q.items() if v is not None}
        menciones = {e: [q.pop(e)] if isinstance(q[e], str) else list(q.pop(e)) for e in CABEZAS_EJE if e in q}
        ejes = set(menciones) | {e for e, a in activa.items() if a}

        def abre(s: dict[str, Any], e: str) -> bool:
            if e == "region":   # con actividad, los cuadros de cada región; sin ella, el cuadro por región
                return s["region_fija"] is not None if "actividad" in ejes else s["apertura"] == "region"
            return s["apertura"] == e

        def ok(s: dict[str, Any]) -> bool:
            for e in CABEZAS_EJE:
                if activa[e]:
                    if not abre(s, e):
                        return False
                elif e in menciones:
                    if s[e] not in menciones[e]:
                        return False
                elif s[e] not in _AGREGADO:
                    return False
            return True

        base = self._base(q)
        r = self._resolver(Catalog([s for s in base.filter(**q) if ok(s)]), {**q, **menciones}, cortes=CORTES)
        r.modo = "cuadro" if any(activa.values()) else "categorias" if ejes else "agregada"
        if r.modo == "cuadro":
            r.destacadas = {e: m for e, m in menciones.items() if activa[e]}
            if r.estado == "resuelto":   # se expande el eje de cada cabeza; los demás conservan su mención
                ids = set(r.cuadros)
                r.candidatos = Catalog([s for s in base.filter(**q, **r.supuestos) if s["cuadro_id"] in ids and
                                        all(s[e] in m for e, m in menciones.items() if not activa[e])])
        return r

    def _resolver(self, cands: "Catalog", q: dict[str, Any], cortes: Iterable[str] = ()) -> Resolucion:
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
        dims = [d for d in DIMS_CUADRO if d not in set(cortes)]
        if len(cands.cuadros()) == 1 or (cortes and cands.widest_dim(dims) is None):
            return Resolucion("resuelto", q, cands, colapsadas, supuestos)
        return Resolucion("aclarar", q, cands, colapsadas, supuestos, aclarar=cands.widest_dim(dims))
