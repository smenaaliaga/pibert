"""
Evaluación detallada del extractor sobre test: F1 por entidad, normalización de CALCULO,
celda del Cuadro 2 por eje y exactitud de marco.

    python util/eval_extractor.py [model_out/pibert_v5] [pibimacecv5]
"""
import json
import sys
from collections import Counter

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from seqeval.metrics import classification_report

from normalizacion import normalizar_calculo
from predict_cli import Predictor

MODEL = sys.argv[1] if len(sys.argv) > 1 else "model_out/pibert_v5"
D = f"data/{sys.argv[2] if len(sys.argv) > 2 else 'pibimacecv5'}"
texts = open(f"{D}/test/seq.in", encoding="utf-8").read().splitlines()
gold_tags = [l.split() for l in open(f"{D}/test/seq.out", encoding="utf-8").read().splitlines()]
gold_heads = [l.split(",") for l in open(f"{D}/test/label", encoding="utf-8").read().splitlines()]
xl = pd.read_excel(f"{D}/raw/dataset.xlsx")
xl = xl[xl.Split == "test"].reset_index(drop=True)
assert [str(t).strip() for t in xl.Utterance] == texts
intended = list(xl.calculo.astype(str))
revisar = list(xl.Revisar.astype(int))

p = Predictor(MODEL)
preds = [p.predict(t) for t in texts]
pred_tags = [r["slot_tags"] for r in preds]
H = ["activ_cls", "reg_cls", "gast_cls"]
pred_heads = [[r["heads"][h]["label"] for h in H] for r in preds]

out = {"n_test": len(texts)}
rep = classification_report(gold_tags, pred_tags, output_dict=True, digits=4)
out["entidades"] = {k: {m: round(v[m], 4) for m in ("precision", "recall", "f1-score", "support")}
                    for k, v in rep.items()}


def ents(words, tags):
    return Predictor.entities_from_bio(words, tags)


cal_gold, cal_pred = [], []
for t, g, pt in zip(texts, gold_tags, pred_tags):
    w = t.split()
    cal_gold.append(normalizar_calculo(ents(w, g).get("CALCULO", [])))
    cal_pred.append(normalizar_calculo(ents(w, pt).get("CALCULO", [])))
out["calculo_vs_spans_gold"] = sum(a == b for a, b in zip(cal_gold, cal_pred)) / len(texts)
out["calculo_vs_intencion"] = sum(a == b for a, b in zip(intended, cal_pred)) / len(texts)
ok_rows = [i for i, r in enumerate(revisar) if not r]
out["calculo_vs_intencion_sin_revisar"] = sum(intended[i] == cal_pred[i] for i in ok_rows) / len(ok_rows)
out["calculo_confusion"] = {f"{a}->{b}": n for (a, b), n in Counter(zip(cal_gold, cal_pred)).items()}

EJE = {"activ_cls": "ACTIVIDAD", "reg_cls": "REGION", "gast_cls": "GASTO"}
celda_ok = Counter()
celdas_gold = {h: Counter() for h in H}
joint = 0
for t, g, pt, gh, ph in zip(texts, gold_tags, pred_tags, gold_heads, pred_heads):
    w = t.split()
    eg, ep = ents(w, g), ents(w, pt)
    allok = True
    for i, h in enumerate(H):
        cg = (EJE[h] in eg, gh[i])
        cp = (EJE[h] in ep, ph[i])
        celdas_gold[h][f"span={'si' if cg[0] else '-'},cabeza={cg[1]}"] += 1
        celda_ok[h] += cg == cp
        allok &= cg == cp
    joint += allok
out["cuadro2_celda_acc"] = {h: celda_ok[h] / len(texts) for h in H}
out["cuadro2_celda_acc_conjunta"] = joint / len(texts)
out["cuadro2_distribucion_gold"] = {h: dict(c) for h, c in celdas_gold.items()}
out["frame_exacto"] = sum(g == pt and gh == ph for g, pt, gh, ph in
                          zip(gold_tags, pred_tags, gold_heads, pred_heads)) / len(texts)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
