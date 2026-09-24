"""
JointBERT dirigido por esquema: un encoder Transformer, una cabeza de clasificación por
entrada de `schema.heads` (sobre el token [CLS]) y una cabeza de secuencia para slots BIO
(softmax o CRF). Nada aquí nombra una cabeza concreta.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel

from schema import Schema
from .module import Classifier

try:
    from torchcrf import CRF
except ImportError:  # pragma: no cover
    CRF = None


@dataclass
class JointOutput:
    loss: torch.Tensor | None
    head_logits: dict[str, torch.Tensor]
    slot_logits: torch.Tensor


class JointBERT(nn.Module):
    def __init__(self, schema: Schema, model_name_or_path: str | None = None, config=None,
                 dropout_rate: float = 0.1, use_crf: bool = False, slot_loss_coef: float = 1.0,
                 ignore_index: int = -100):
        super().__init__()
        self.schema = schema
        self.use_crf = use_crf
        self.slot_loss_coef = slot_loss_coef
        self.ignore_index = ignore_index
        self.dropout_rate = dropout_rate

        if model_name_or_path is not None:
            self.encoder = AutoModel.from_pretrained(model_name_or_path, config=config)
        else:
            self.encoder = AutoModel.from_config(config)
        self.config = self.encoder.config
        hidden = self.config.hidden_size
        self._accepts_token_type = "token_type_ids" in inspect.signature(self.encoder.forward).parameters

        self.heads = nn.ModuleDict({name: Classifier(hidden, len(labels), dropout_rate)
                                    for name, labels in schema.heads.items()})
        self.slot_classifier = Classifier(hidden, len(schema.slot_labels), dropout_rate)
        if use_crf:
            if CRF is None:
                raise ImportError("pytorch-crf no está instalado: pip install pytorch-crf")
            self.crf = CRF(len(schema.slot_labels), batch_first=True)

    # ---- forward ---------------------------------------------------------------
    def forward(self, input_ids, attention_mask, token_type_ids=None,
                head_label_ids: torch.Tensor | None = None, slot_labels_ids=None) -> JointOutput:
        kwargs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self._accepts_token_type and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        sequence_output = self.encoder(**kwargs)[0]
        pooled = sequence_output[:, 0]

        head_logits = {name: head(pooled) for name, head in self.heads.items()}
        slot_logits = self.slot_classifier(sequence_output)

        loss = None
        if head_label_ids is not None:
            loss = sequence_output.new_zeros(())
            ce = nn.CrossEntropyLoss()
            for i, name in enumerate(self.schema.head_names):
                loss = loss + ce(head_logits[name], head_label_ids[:, i])
        if slot_labels_ids is not None and self.slot_loss_coef != 0:
            if self.use_crf:
                tags = slot_labels_ids.clone()
                tags[tags == self.ignore_index] = 0  # el CRF no admite ignore_index; O en posiciones ignoradas
                slot_loss = -self.crf(slot_logits, tags, mask=attention_mask.bool(), reduction="mean")
            else:
                ce_slot = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
                active = attention_mask.view(-1) == 1
                slot_loss = ce_slot(slot_logits.view(-1, len(self.schema.slot_labels))[active],
                                    slot_labels_ids.view(-1)[active])
            loss = (loss if loss is not None else 0) + self.slot_loss_coef * slot_loss
        return JointOutput(loss, head_logits, slot_logits)

    def decode_slots(self, slot_logits: torch.Tensor, attention_mask: torch.Tensor) -> list[list[int]]:
        if self.use_crf:
            return self.crf.decode(slot_logits, mask=attention_mask.bool())
        return slot_logits.argmax(dim=-1).tolist()

    # ---- persistencia ------------------------------------------------------------
    def save(self, out_dir: Path | str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(out)
        save_file({k: v.contiguous() for k, v in self.state_dict().items()}, out / "model.safetensors")
        self.schema.save(out)
        (out / "jointbert.json").write_text(json.dumps({
            "dropout_rate": self.dropout_rate, "use_crf": self.use_crf,
            "slot_loss_coef": self.slot_loss_coef, "ignore_index": self.ignore_index,
        }, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: Path | str, device: str | torch.device = "cpu") -> "JointBERT":
        d = Path(model_dir)
        schema = Schema.load(d)
        cfg = json.loads((d / "jointbert.json").read_text(encoding="utf-8"))
        model = cls(schema, model_name_or_path=None, config=AutoConfig.from_pretrained(d), **cfg)
        model.load_state_dict(load_file(d / "model.safetensors"))
        return model.to(device).eval()
