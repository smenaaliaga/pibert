"""
Migra pibimacecv4 (3 cabezas binarias + calculo, entidad REFERENCIA) a pibimacecv5, la
definición del paper (sección 4): 3 cabezas binarias y CALCULO como entidad.

Transformación sobre data/<src>/raw/dataset.xlsx:
  REFERENCIA            sus tags pasan a O: la referencia se resuelve fuera del modelo
  FRECUENCIA            "interanual(mente)" y "mensual o trimestral" se re-etiquetan como CALCULO
                        (nombran el cálculo, no la periodicidad)
  CALCULO               pre-anotación por léxico (normalizacion.spans_calculo, sólo sobre tokens O)
  calculo               deja de ser cabeza; la columna se conserva como el valor esperado de la
                        normalización de los spans. Si normalizar_calculo(spans) no coincide, la fila
                        se marca Revisar con su Motivo.

Escribe data/<dst>/raw/dataset.xlsx, schema.json y, con util/process_raw_data.py, processed/ y
train/dev/test según la columna Split.

Uso:
    python util/migrate_v4_to_v5.py [--src pibimacecv4] [--dst pibimacecv5]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalizacion import norm, normalizar_calculo, spans_calculo  # noqa: E402
from process_raw_data import process_file  # noqa: E402
from schema import Schema  # noqa: E402


def frecuencia_es_calculo(words: list[str], tags: list[str]) -> list[int]:
    """Índices etiquetados FRECUENCIA que en realidad nombran el cálculo."""
    out = []
    for i, (w, t) in enumerate(zip(words, tags)):
        if t.endswith("-FRECUENCIA") and norm(w).startswith("interanual"):
            out.append(i)
        if (t == "B-FRECUENCIA" and i + 2 < len(words) and norm(words[i + 1]) == "o"
                and tags[i + 2] == "B-FRECUENCIA"
                and {norm(words[i]), norm(words[i + 2])} == {"mensual", "trimestral"}):
            out += [i, i + 1, i + 2]
    return out


def migrate_row(text: str, tags_v4: str, calculo: str, stats: Counter) -> tuple[str, str]:
    words, tags = text.split(), tags_v4.split()
    tags = ["O" if t.endswith("-REFERENCIA") else t for t in tags]
    if tags != tags_v4.split():
        stats["referencia_a_O"] += 1
    retag = frecuencia_es_calculo(words, tags)
    for i in retag:
        tags[i] = "O"
    if retag:
        stats["frecuencia_a_calculo"] += 1

    spans = spans_calculo(words, [t == "O" for t in tags])
    for a, b, _ in spans:
        tags[a:b] = ["B-CALCULO"] + ["I-CALCULO"] * (b - a - 1)
    stats[f"spans_calculo={len(spans)}"] += 1

    normalizado = normalizar_calculo([" ".join(words[a:b]) for a, b, _ in spans])
    if normalizado == calculo:
        return " ".join(tags), ""
    stats[f"calculo {calculo} -> {normalizado}"] += 1
    if normalizado == "nivel":
        return " ".join(tags), f"calculo={calculo} sin mención CALCULO"
    return " ".join(tags), f"calculo={calculo}, los spans normalizan a {normalizado}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="pibimacecv4")
    p.add_argument("--dst", default="pibimacecv5")
    p.add_argument("--data_dir", default="data")
    a = p.parse_args()

    src, dst = Path(a.data_dir) / a.src, Path(a.data_dir) / a.dst
    df = pd.read_excel(src / "raw" / "dataset.xlsx")
    schema = Schema.from_catalog()
    (dst / "raw").mkdir(parents=True, exist_ok=True)
    schema.save(dst)

    stats: Counter = Counter()
    nuevos, motivos = [], []
    for text, tags, calculo in zip(df["Utterance"], df["NER_BIO"], df["calculo"]):
        t, m = migrate_row(str(text).strip(), str(tags).strip(), str(calculo).strip(), stats)
        nuevos.append(t)
        motivos.append(m)
    df["NER_BIO"] = nuevos
    df["Motivo"] = motivos
    df["Revisar"] = [int(bool(r) or bool(m)) for r, m in zip(df["Revisar"], motivos)]
    for t in (t for tags in nuevos for t in tags.split()):
        schema.slot_index(t)   # validación estricta contra el esquema
    cols = ["Utterance", *schema.head_names, "NER_BIO", "calculo", "Split", "Aumentada", "Revisar", "Motivo"]
    df[cols].to_excel(dst / "raw" / "dataset.xlsx", index=False)

    process_file(a.dst, str(dst / "raw" / "dataset.xlsx"))

    ents = Counter(t[2:] for tags in nuevos for t in tags.split() if t.startswith("B-"))
    report = [f"# Migración {a.src} → {a.dst}", "",
              f"- filas: {len(df)}; por split: {dict(Counter(df['Split']))}",
              f"- esquema: cabezas {schema.head_names}; entidades {schema.entities}",
              f"- spans por entidad: {dict(ents.most_common())}",
              f"- transformaciones: {dict(sorted(stats.items()))}",
              f"- filas para revisar a mano: {int(df['Revisar'].sum())} (columna Revisar; Motivo para las nuevas)", ""]
    nuevas = [(m, t) for m, t in zip(motivos, df["Utterance"]) if m]
    if nuevas:
        report += [f"## Cálculo sin mención consistente: {len(nuevas)}", ""]
        report += [f"- [{m}] {t}" for m, t in sorted(nuevas)] + [""]
    (dst / "migracion_reporte.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(l for l in report if not l.startswith("- [")))


if __name__ == "__main__":
    main()
