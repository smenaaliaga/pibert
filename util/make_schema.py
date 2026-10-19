"""
Genera data/<task>/schema.json.

    python util/make_schema.py --task pibimacecv3 --from-data          # esquema heredado (5 cabezas)
    python util/make_schema.py --task pibimacecv5 --from-catalog       # esquema del paper (3 binarias, CALCULO entidad)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema import CABEZAS_LEGADO, Schema

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--data_dir", default="data")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-data", action="store_true")
    g.add_argument("--from-catalog", action="store_true")
    p.add_argument("--heads", default=",".join(CABEZAS_LEGADO), help="orden de las cabezas en `label` (--from-data)")
    p.add_argument("--vocab", default="data/catalogo/vocabularios.json")
    a = p.parse_args()

    out = Path(a.data_dir) / a.task
    out.mkdir(parents=True, exist_ok=True)
    s = Schema.from_split_files(out, a.heads.split(",")) if a.from_data else Schema.from_catalog(a.vocab)
    path = s.save(out)
    print(f"{path}: {len(s.heads)} cabezas {dict((h, len(l)) for h, l in s.heads.items())}, "
          f"{len(s.entities)} entidades {s.entities}, {len(s.slot_labels)} etiquetas BIO")
