"""
Lectura de data/<task>/{train,dev,test}/{seq.in,label,seq.out} y conversión a tensores.

`label` trae una etiqueta por cabeza, separadas por coma, en el orden de schema.heads.
Cualquier etiqueta o tag BIO fuera del esquema es un error: nunca se mapea en silencio.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset

from schema import Schema

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    guid: str
    words: list[str]
    head_labels: list[int]      # un índice por cabeza, en orden de schema.heads
    slot_labels: list[int]      # un índice por palabra


@dataclass
class InputFeatures:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    head_label_ids: list[int]
    slot_labels_ids: list[int]


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def read_examples(data_dir: str, task: str, mode: str, schema: Schema) -> list[InputExample]:
    d = os.path.join(data_dir, task, mode)
    logger.info("LOOKING AT %s", d)
    texts = _read_lines(os.path.join(d, "seq.in"))
    labels = _read_lines(os.path.join(d, "label"))
    slots = _read_lines(os.path.join(d, "seq.out"))
    if not (len(texts) == len(labels) == len(slots)):
        raise ValueError(f"{d}: seq.in/label/seq.out tienen distinto número de líneas "
                         f"({len(texts)}/{len(labels)}/{len(slots)})")
    heads = schema.head_names
    examples = []
    for i, (text, label, slot) in enumerate(zip(texts, labels, slots), 1):
        words = text.split()
        tags = slot.split()
        if len(words) != len(tags):
            raise ValueError(f"{d}/seq.out:{i}: {len(tags)} etiquetas para {len(words)} palabras: {text!r}")
        parts = [p.strip() for p in label.split(",")]
        if len(parts) != len(heads):
            raise ValueError(f"{d}/label:{i}: {len(parts)} valores, el esquema tiene {len(heads)} cabezas {heads}")
        try:
            head_labels = [schema.head_index(h, v) for h, v in zip(heads, parts)]
            slot_labels = [schema.slot_index(t) for t in tags]
        except ValueError as e:
            raise ValueError(f"{d}:{i}: {e}") from None
        examples.append(InputExample(f"{mode}-{i}", words, head_labels, slot_labels))
    return examples


def convert_examples_to_features(examples: list[InputExample], max_seq_len: int, tokenizer,
                                 pad_token_label_id: int = -100) -> list[InputFeatures]:
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    unk = tokenizer.unk_token
    features = []
    truncated = 0
    for ex in examples:
        tokens: list[str] = []
        slot_ids: list[int] = []
        for word, slot in zip(ex.words, ex.slot_labels):
            wt = tokenizer.tokenize(word) or [unk]
            tokens.extend(wt)
            # etiqueta real en el primer subtoken, ignorada en el resto
            slot_ids.extend([slot] + [pad_token_label_id] * (len(wt) - 1))
        if len(tokens) > max_seq_len - 2:
            truncated += 1
            tokens, slot_ids = tokens[: max_seq_len - 2], slot_ids[: max_seq_len - 2]
        input_ids = [cls_id] + tokenizer.convert_tokens_to_ids(tokens) + [sep_id]
        slot_ids = [pad_token_label_id] + slot_ids + [pad_token_label_id]
        pad = max_seq_len - len(input_ids)
        features.append(InputFeatures(
            input_ids=input_ids + [pad_id] * pad,
            attention_mask=[1] * len(input_ids) + [0] * pad,
            token_type_ids=[0] * max_seq_len,
            head_label_ids=ex.head_labels,
            slot_labels_ids=slot_ids + [pad_token_label_id] * pad,
        ))
    if truncated:
        logger.warning("%d ejemplos truncados a max_seq_len=%d", truncated, max_seq_len)
    return features


def load_and_cache_examples(args, tokenizer, schema: Schema, mode: str) -> TensorDataset:
    model_tag = [p for p in str(args.model_name_or_path).split("/") if p][-1]
    # el contenido de los datos entra en la clave: editar el Excel y reprocesar invalida el caché
    h = hashlib.sha1()
    for name in ("seq.in", "label", "seq.out"):
        with open(os.path.join(args.data_dir, args.task, mode, name), "rb") as f:
            h.update(f.read())
    cached = os.path.join(args.data_dir, args.task,
                          f"cached_{mode}_{model_tag}_{args.max_seq_len}_{schema.fingerprint()}_{h.hexdigest()[:10]}")
    if os.path.exists(cached):
        logger.info("Loading features from cached file %s", cached)
        features = torch.load(cached, weights_only=False)
    else:
        examples = read_examples(args.data_dir, args.task, mode, schema)
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, args.ignore_index)
        torch.save(features, cached)
    return TensorDataset(
        torch.tensor([f.input_ids for f in features], dtype=torch.long),
        torch.tensor([f.attention_mask for f in features], dtype=torch.long),
        torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
        torch.tensor([f.head_label_ids for f in features], dtype=torch.long),
        torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long),
    )
