from __future__ import annotations

import logging
import random

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import f1_score as sk_f1
from transformers import AutoTokenizer

from schema import Schema

MODEL_PATH_MAP = {
    "bert": "bert-base-uncased",
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
    "deberta": "microsoft/mdeberta-v3-base",
    "roberta": "bertin-project/bertin-roberta-base-spanish",
    "tiny": "google/bert_uncased_L-2_H-128_A-2",   # sólo para pruebas de humo
}


def resolve_model_name(model_type: str) -> str:
    """Atajo de MODEL_PATH_MAP o un identificador/ruta de Hugging Face tal cual."""
    return MODEL_PATH_MAP.get(model_type, model_type)


def load_tokenizer(name_or_path: str):
    return AutoTokenizer.from_pretrained(name_or_path)


def init_logger():
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)


def pick_device(no_cuda: bool = False) -> str:
    if not no_cuda and torch.cuda.is_available():
        return "cuda"
    if not no_cuda and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int, device: str = "cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device == "mps":
        torch.mps.manual_seed(seed)


def compute_metrics(schema: Schema, head_preds: dict[str, np.ndarray], head_labels: dict[str, np.ndarray],
                    slot_preds: list[list[str]], slot_labels: list[list[str]]) -> dict[str, float]:
    """Accuracy y macro-F1 por cabeza, P/R/F1 de slots (seqeval) y exactitud de frame semántico."""
    results: dict[str, float] = {}
    frame_ok = np.ones(len(slot_labels), dtype=bool)
    for name in schema.head_names:
        p, l = head_preds[name], head_labels[name]
        results[f"{name}_acc"] = float((p == l).mean())
        results[f"{name}_f1"] = float(sk_f1(l, p, average="macro", zero_division=0))
        frame_ok &= p == l
    results["slot_precision"] = float(precision_score(slot_labels, slot_preds))
    results["slot_recall"] = float(recall_score(slot_labels, slot_preds))
    results["slot_f1"] = float(f1_score(slot_labels, slot_preds))
    slot_ok = np.array([p == l for p, l in zip(slot_preds, slot_labels)])
    results["semantic_frame_acc"] = float((frame_ok & slot_ok).mean())
    return results
