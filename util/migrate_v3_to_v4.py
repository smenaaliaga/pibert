"""
Migra pibimacecv3 (esquema heredado: 5 cabezas, 7 entidades) a pibimacecv4 (esquema del
catálogo: 3 cabezas binarias + calculo, 10 entidades).

Transformación mecánica:
  activity/region/investment  general -> 1, specific/none -> 0     (activ_cls, reg_cls, gast_cls)
  calc_mode                   original -> nivel, yoy -> var_anual, prev_period -> var_periodo_anterior,
                              contribution -> nivel (la contribución pasa a ser un span PRESENTACION)
  req_form                    se elimina (se deriva del normalizador de PERIODO)
  entidades                   INDICATOR->INDICADOR, PERIOD->PERIODO, SEASONALITY->ESTACIONALIDAD,
                              FREQUENCY->FRECUENCIA, ACTIVITY->ACTIVIDAD, INVESTMENT->GASTO

Pre-anotación por léxico (sólo sobre tokens O, nunca pisa un span existente):
  PRESENTACION  contribución / incidencia / aporte / peso / participación y sus verbos
Aumento (los datos v3 no mencionan valoración ni referencia):
  VALORACION, REFERENCIA  se añaden frases a una fracción de las consultas de PIB, con sus tags

Uso:
    python util/migrate_v3_to_v4.py [--src pibimacecv3] [--dst pibimacecv4] [--aug-val 0.06] [--aug-ref 0.03]
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema import Schema  # noqa: E402

HEAD_ORDER_V3 = ["calc_mode", "activity", "region", "investment", "req_form"]
CALCULO = {"original": "nivel", "yoy": "var_anual", "prev_period": "var_periodo_anterior", "contribution": "nivel"}
ENTIDAD = {"INDICATOR": "INDICADOR", "PERIOD": "PERIODO", "SEASONALITY": "ESTACIONALIDAD",
           "FREQUENCY": "FRECUENCIA", "ACTIVITY": "ACTIVIDAD", "REGION": "REGION", "INVESTMENT": "GASTO"}

# léxico de PRESENTACION: una palabra basta como span (incidió, contribuyó, aportó, peso, participación)
PRESENTACION_RE = re.compile(r"^(contribu\w*|incid\w*|aport\w*|peso|particip\w*|explic\w*|impuls\w*)$")

FRASES_VALORACION = ["a precios corrientes", "en términos nominales", "nominal", "en pesos corrientes",
                     "en volumen", "en términos reales", "real", "encadenado", "a precios encadenados",
                     "a precios constantes"]
FRASES_REFERENCIA = ["referencia 2013", "referencia 2008", "base 2003", "compilación de referencia 2013",
                     "de la referencia 2018", "base 1996"]
PIB_RE = re.compile(r"\b(pib|producto interno bruto|pibr)\b", re.I)
REVISAR: list[tuple[str, str]] = []   # filas a revisar a mano


def norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def read_split(root: Path, split: str) -> list[tuple[str, str, str]]:
    d = root / split
    rows = list(zip(*[(d / f).read_text(encoding="utf-8").splitlines() for f in ("seq.in", "label", "seq.out")]))
    return [(t.replace("\xa0", " ").strip(), l, o) for t, l, o in rows]


def migrate_row(text: str, label: str, tags: str, stats: Counter) -> tuple[str, str, str] | None:
    v3 = dict(zip(HEAD_ORDER_V3, [p.strip() for p in label.split(",")]))
    words, ts = text.split(), tags.split()
    if len(words) != len(ts):
        stats["descartado_desalineado"] += 1
        return None
    ts = [t if t == "O" else f"{t[:2]}{ENTIDAD[t[2:]]}" for t in ts]

    if v3["calc_mode"] == "contribution":
        hit = False
        for i, (w, t) in enumerate(zip(words, ts)):
            if t == "O" and PRESENTACION_RE.match(norm(w).strip(",.?¿!¡")):
                ts[i] = "B-PRESENTACION"
                hit = True
                break
        stats["contribution_con_span" if hit else "contribution_sin_span"] += 1
        if not hit:
            REVISAR.append(("contribution sin span PRESENTACION", text))
    if v3["region"] == "specific" and "B-REGION" not in ts:
        stats["region_specific_sin_span"] += 1
        REVISAR.append(("region=specific sin span REGION (queda reg_cls=0)", text))
    heads = [str(int(v3["activity"] == "general")), str(int(v3["region"] == "general")),
             str(int(v3["investment"] == "general")), CALCULO[v3["calc_mode"]]]
    return " ".join(words), ",".join(heads), " ".join(ts)


def augment(rows: list[tuple[str, str, str]], frac: float, frases: list[str], etype: str,
            rng: random.Random) -> list[tuple[str, str, str]]:
    """Añade una frase de `etype` al final de una fracción de las consultas de PIB, con sus tags BIO."""
    cands = [i for i, (t, _, _) in enumerate(rows) if PIB_RE.search(t)]
    out = []
    for i in rng.sample(cands, min(len(cands), int(len(rows) * frac))):
        t, l, o = rows[i]
        frase = rng.choice(frases)
        n = len(frase.split())
        out.append((f"{t} {frase}", l, o + " " + " ".join([f"B-{etype}"] + [f"I-{etype}"] * (n - 1))))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="pibimacecv3")
    p.add_argument("--dst", default="pibimacecv4")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--aug-val", type=float, default=0.06, help="fracción de filas aumentadas con VALORACION")
    p.add_argument("--aug-ref", type=float, default=0.03, help="fracción de filas aumentadas con REFERENCIA")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    src, dst = Path(a.data_dir) / a.src, Path(a.data_dir) / a.dst
    schema = Schema.from_catalog()
    schema.save(dst.mkdir(parents=True, exist_ok=True) or dst)
    rng = random.Random(a.seed)
    report = ["# Migración v3 → v4", ""]
    for split in ("train", "dev", "test"):
        stats: Counter = Counter()
        rows = [r for r in (migrate_row(*row, stats) for row in read_split(src, split)) if r]
        aug = augment(rows, a.aug_val, FRASES_VALORACION, "VALORACION", rng) + \
              augment(rows, a.aug_ref, FRASES_REFERENCIA, "REFERENCIA", rng)
        rows += aug
        rng.shuffle(rows)
        d = dst / split
        d.mkdir(exist_ok=True)
        for name, idx in (("seq.in", 0), ("label", 1), ("seq.out", 2)):
            (d / name).write_text("\n".join(r[idx] for r in rows) + "\n", encoding="utf-8")
        # validación estricta contra el esquema
        for r in rows:
            for h, v in zip(schema.head_names, r[1].split(",")):
                schema.head_index(h, v)
            for t in r[2].split():
                schema.slot_index(t)
        heads = Counter()
        for r in rows:
            for h, v in zip(schema.head_names, r[1].split(",")):
                heads[f"{h}={v}"] += 1
        ents = Counter(t[2:] for r in rows for t in r[2].split() if t.startswith("B-"))
        report += [f"## {split}: {len(rows)} filas ({len(aug)} aumentadas)", "",
                   f"- cabezas: {dict(sorted(heads.items()))}", f"- spans por entidad: {dict(ents.most_common())}",
                   f"- avisos: {dict(stats)}", ""]
    if REVISAR:
        report += [f"## Filas para revisar a mano: {len(REVISAR)}", ""]
        report += [f"- [{motivo}] {texto}" for motivo, texto in REVISAR] + [""]
    (dst / "migracion_reporte.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(l for l in report if not l.startswith("- [")))


if __name__ == "__main__":
    main()
