"""
Esquema del modelo: qué cabezas clasificadoras y qué tipos de entidad produce PIBERT.

El esquema es la única fuente de verdad para model_in/, data_loader.py, trainer.py y
predict_cli.py: agregar o quitar una cabeza es cambiar el esquema, no el código.

Dos orígenes:
  - Schema.from_catalog(vocabularios.json)      la definición nueva (3 cabezas binarias + calculo,
                                                 10 tipos de entidad)
  - Schema.from_split_files(data/<task>, heads)  la definición heredada de un dataset ya etiquetado

Se persiste como schema.json junto a los datos y junto al checkpoint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

# Definición nueva (paper, sección 4): cabezas y tipos de entidad
CABEZAS_CATALOGO: dict[str, list[str]] = {
    "activ_cls": ["0", "1"],
    "reg_cls": ["0", "1"],
    "gast_cls": ["0", "1"],
    "calculo": ["nivel", "var_anual", "var_periodo_anterior"],
}
ENTIDADES_SOLICITUD = ["PERIODO", "REFERENCIA"]

# Cabezas del dataset heredado (pibimacecv3), en el orden de la columna `label`
CABEZAS_LEGADO = ["calc_mode", "activity", "region", "investment", "req_form"]


@dataclass
class Schema:
    heads: dict[str, list[str]]
    entities: list[str]
    slot_labels: list[str] = field(default_factory=list)
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.slot_labels:
            self.slot_labels = ["O"] + [f"{p}-{e}" for e in self.entities for p in ("B", "I")]
        if self.slot_labels[0] != "O":
            raise ValueError("slot_labels debe empezar con 'O'")
        self._head_idx = {h: {l: i for i, l in enumerate(ls)} for h, ls in self.heads.items()}
        self._slot_idx = {t: i for i, t in enumerate(self.slot_labels)}

    # ---- consultas -----------------------------------------------------------
    @property
    def head_names(self) -> list[str]:
        return list(self.heads)

    def num_labels(self, head: str) -> int:
        return len(self.heads[head])

    def head_index(self, head: str, label: str) -> int:
        try:
            return self._head_idx[head][label]
        except KeyError:
            raise ValueError(f"etiqueta desconocida {label!r} para la cabeza {head!r}; "
                             f"válidas: {self.heads[head]}") from None

    def slot_index(self, tag: str) -> int:
        try:
            return self._slot_idx[tag]
        except KeyError:
            raise ValueError(f"etiqueta BIO desconocida {tag!r}; válidas: {self.slot_labels}") from None

    def fingerprint(self) -> str:
        import hashlib
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]

    # ---- persistencia ------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"version": self.version, "heads": self.heads,
                "entities": self.entities, "slot_labels": self.slot_labels}

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        if path.is_dir():
            path = path / "schema.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Schema":
        path = Path(path)
        if path.is_dir():
            path = path / "schema.json"
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(heads=d["heads"], entities=d["entities"], slot_labels=d["slot_labels"],
                   version=str(d.get("version", "1")))

    # ---- constructores ---------------------------------------------------------
    @classmethod
    def from_catalog(cls, vocab_path: Path | str = "data/catalogo/vocabularios.json") -> "Schema":
        voc = json.loads(Path(vocab_path).read_text(encoding="utf-8"))["modelo"]
        entities = list(voc["entidades"]) + ENTIDADES_SOLICITUD
        return cls(heads=dict(CABEZAS_CATALOGO), entities=entities, version="catalogo")

    @classmethod
    def from_split_files(cls, data_dir: Path | str, head_names: Iterable[str] = CABEZAS_LEGADO,
                         splits: Iterable[str] = ("train", "dev", "test")) -> "Schema":
        """Deriva el esquema de un dataset ya etiquetado (label con cabezas separadas por coma, seq.out BIO)."""
        data_dir = Path(data_dir)
        head_names = list(head_names)
        labels: dict[str, set[str]] = {h: set() for h in head_names}
        tags: set[str] = set()
        for split in splits:
            d = data_dir / split
            if not d.exists():
                continue
            for ln, line in enumerate((d / "label").read_text(encoding="utf-8").splitlines(), 1):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != len(head_names):
                    raise ValueError(f"{d/'label'}:{ln}: {len(parts)} valores, se esperaban {len(head_names)} ({head_names})")
                for h, v in zip(head_names, parts):
                    labels[h].add(v)
            for line in (d / "seq.out").read_text(encoding="utf-8").splitlines():
                tags.update(line.split())
        entities = sorted({t[2:] for t in tags if t != "O"})
        slot_labels = ["O"] + [f"{p}-{e}" for e in entities for p in ("B", "I")]
        extra = sorted(tags - set(slot_labels))
        if extra:
            raise ValueError(f"etiquetas BIO fuera del esquema B-/I-/O: {extra}")
        return cls(heads={h: sorted(labels[h]) for h in head_names}, entities=entities,
                   slot_labels=slot_labels, version="legado")
