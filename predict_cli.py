"""
Inferencia desde línea de comandos.

    python predict_cli.py --model_dir model_out/pibert                      # interactivo
    python predict_cli.py --model_dir model_out/pibert --text "cual fue el ultimo imacec"
    python predict_cli.py --model_dir model_out/pibert --input_file consultas.txt --output_file salida.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model_in import JointBERT
from utils import init_logger, pick_device

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_dir: str, no_cuda: bool = False):
        self.device = pick_device(no_cuda)
        self.model = JointBERT.load(model_dir, device=self.device)
        self.schema = self.model.schema
        self.args = json.loads((Path(model_dir) / "training_args.json").read_text(encoding="utf-8"))
        self.max_seq_len = int(self.args.get("max_seq_len", 50))
        # el tokenizer se guarda junto al checkpoint al terminar el entrenamiento; el hub sólo como respaldo.
        # Sin tokenizer_config.json, from_pretrained(model_dir) no falla: arma un tokenizer sin vocabulario.
        if (Path(model_dir) / "tokenizer_config.json").exists():
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            logger.warning("Tokenizer no encontrado en %s; se carga %s", model_dir, self.args["model_name_or_path"])
            self.tokenizer = AutoTokenizer.from_pretrained(self.args["model_name_or_path"])
        logger.info("Model loaded from %s (%s)", model_dir, self.device)

    @staticmethod
    def entities_from_bio(words: list[str], tags: list[str]) -> dict[str, list[str]]:
        spans, cur_type, cur = [], None, []
        for w, t in zip(words, tags):
            prefix, _, etype = t.partition("-")
            if t == "O" or not etype:
                if cur:
                    spans.append((cur_type, " ".join(cur)))
                cur_type, cur = None, []
            elif prefix == "I" and cur_type == etype:
                cur.append(w)
            else:
                if cur:
                    spans.append((cur_type, " ".join(cur)))
                cur_type, cur = etype, [w]
        if cur:
            spans.append((cur_type, " ".join(cur)))
        grouped = defaultdict(list)
        for etype, text in spans:
            grouped[etype].append(text)
        return dict(grouped)

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        words = text.split()
        tokens, word_ids = [], []
        for i, w in enumerate(words):
            wt = self.tokenizer.tokenize(w) or [self.tokenizer.unk_token]
            tokens.extend(wt)
            word_ids.extend([i] * len(wt))
        tokens, word_ids = tokens[: self.max_seq_len - 2], word_ids[: self.max_seq_len - 2]
        ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.sep_token_id]
        word_ids = [-1] + word_ids + [-1]
        pad = self.max_seq_len - len(ids)
        input_ids = torch.tensor([ids + [self.tokenizer.pad_token_id] * pad], device=self.device)
        attention = torch.tensor([[1] * len(ids) + [0] * pad], device=self.device)
        token_type = torch.zeros_like(input_ids)

        out = self.model(input_ids, attention, token_type)
        heads = {}
        for name, logits in out.head_logits.items():
            probs = torch.softmax(logits[0], dim=-1)
            idx = int(probs.argmax())
            heads[name] = {"label": self.schema.heads[name][idx], "confidence": float(probs[idx])}
        pred_ids = self.model.decode_slots(out.slot_logits, attention)[0]
        tags = ["O"] * len(words)
        seen = set()
        for pos, wi in enumerate(word_ids):
            if wi >= 0 and wi not in seen and pos < len(pred_ids):
                seen.add(wi)
                tags[wi] = self.schema.slot_labels[pred_ids[pos]]
        return {"text": text, "words": words, "heads": heads, "slot_tags": tags,
                "entities": self.entities_from_bio(words, tags)}

    @staticmethod
    def format_result(r: dict) -> str:
        lines = ["=" * 60]
        for name, h in r["heads"].items():
            lines.append(f"{name:<12}: {h['label']:<22} ({h['confidence']:.1%})")
        lines.append("-" * 60)
        if r["entities"]:
            for etype, mentions in r["entities"].items():
                lines.append(f"  {etype}: {mentions}")
        else:
            lines.append("  (sin entidades)")
        lines.append("=" * 60)
        return "\n".join(lines)


def main():
    init_logger()
    p = argparse.ArgumentParser(description="Predicción CLI para PIBERT")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--text")
    p.add_argument("--input_file")
    p.add_argument("--output_file", help="salida JSONL (una predicción por línea)")
    p.add_argument("--json", action="store_true", help="imprime JSON en vez del formato legible")
    p.add_argument("--no_cuda", action="store_true")
    a = p.parse_args()

    predictor = Predictor(a.model_dir, a.no_cuda)
    show = (lambda r: print(json.dumps(r, ensure_ascii=False))) if a.json else (lambda r: print(predictor.format_result(r)))

    if a.text:
        show(predictor.predict(a.text))
    elif a.input_file:
        texts = [l.strip() for l in open(a.input_file, encoding="utf-8") if l.strip()]
        results = [predictor.predict(t) for t in texts]
        if a.output_file:
            with open(a.output_file, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{len(results)} predicciones guardadas en {a.output_file}")
        else:
            for r in results:
                show(r)
    else:
        print("Modo interactivo. Escribe 'salir' para terminar.")
        while True:
            try:
                text = input("Consulta: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not text:
                continue
            if text.lower() in ("salir", "exit", "quit", "q"):
                break
            show(predictor.predict(text))


if __name__ == "__main__":
    main()
