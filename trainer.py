from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup

from model_in import JointBERT
from schema import Schema
from utils import compute_metrics, pick_device

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args, schema: Schema, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.schema = schema
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.device = pick_device(args.no_cuda)
        self.model = JointBERT(schema, model_name_or_path=args.model_name_or_path,
                               dropout_rate=args.dropout_rate, use_crf=args.use_crf,
                               slot_loss_coef=args.slot_loss_coef, ignore_index=args.ignore_index)
        self.model.to(self.device)

    def _inputs(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        return {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                "head_label_ids": batch[3],
                "slot_labels_ids": batch[4] if self.args.slot_loss_coef != 0 else None}

    def train(self):
        loader = DataLoader(self.train_dataset, sampler=RandomSampler(self.train_dataset),
                            batch_size=self.args.train_batch_size)
        steps_per_epoch = max(len(loader) // self.args.gradient_accumulation_steps, 1)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // steps_per_epoch + 1
        else:
            t_total = steps_per_epoch * int(self.args.num_train_epochs)

        no_decay = ("bias", "LayerNorm.weight")
        params = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(params, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.args.warmup_steps, t_total)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d | Epochs = %d | Batch = %d x %d | Total steps = %d",
                    len(self.train_dataset), int(self.args.num_train_epochs), self.args.train_batch_size,
                    self.args.gradient_accumulation_steps, t_total)

        global_step, tr_loss = 0, 0.0
        best_val_loss, epochs_no_improve = float("inf"), 0
        self.model.zero_grad()
        for epoch_idx in trange(int(self.args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(loader, desc="Iteration", leave=False)):
                self.model.train()
                loss = self.model(**self._inputs(batch)).loss
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1
                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate("dev")
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        # con early stopping el mejor checkpoint vive en model_out; el periódico va a model_out/last
                        self.save_model(os.path.join(self.args.model_out, "last") if self.args.early_stopping
                                        else self.args.model_out)
                if 0 < self.args.max_steps <= global_step:
                    break
            if 0 < self.args.max_steps <= global_step:
                break
            if self.args.early_stopping:
                val_loss = self.evaluate("dev")["loss"]
                if val_loss < best_val_loss:
                    best_val_loss, epochs_no_improve = val_loss, 0
                    self.save_model(self.args.model_out)
                    logger.info("Early stopping: mejor val loss %.4f en la época %d", val_loss, epoch_idx + 1)
                else:
                    epochs_no_improve += 1
                    logger.info("Early stopping: %d época(s) sin mejora", epochs_no_improve)
                    if epochs_no_improve >= self.args.early_stopping_patience:
                        logger.info("Early stopping activado")
                        break
        if not self.args.early_stopping:
            self.save_model(self.args.model_out)
        return global_step, tr_loss / max(global_step, 1)

    @torch.no_grad()
    def evaluate(self, mode: str) -> dict[str, float]:
        dataset = {"dev": self.dev_dataset, "test": self.test_dataset}[mode]
        loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=self.args.eval_batch_size)
        logger.info("***** Running evaluation on %s (%d examples) *****", mode, len(dataset))
        self.model.eval()
        eval_loss, n = 0.0, 0
        head_preds = {h: [] for h in self.schema.head_names}
        head_gold = {h: [] for h in self.schema.head_names}
        slot_preds, slot_gold = [], []
        slot_map = {i: t for i, t in enumerate(self.schema.slot_labels)}
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            inputs = self._inputs(batch)
            out = self.model(**inputs)
            eval_loss += out.loss.item()
            n += 1
            for i, h in enumerate(self.schema.head_names):
                head_preds[h].append(out.head_logits[h].argmax(-1).cpu().numpy())
                head_gold[h].append(inputs["head_label_ids"][:, i].cpu().numpy())
            decoded = self.model.decode_slots(out.slot_logits, inputs["attention_mask"])
            gold = batch[4].numpy()
            for row, pred in zip(gold, decoded):
                keep = [j for j, g in enumerate(row) if g != self.args.ignore_index]
                slot_gold.append([slot_map[row[j]] for j in keep])
                slot_preds.append([slot_map[pred[j]] if j < len(pred) else "O" for j in keep])
        results = {"loss": eval_loss / max(n, 1)}
        results.update(compute_metrics(self.schema,
                                       {h: np.concatenate(v) for h, v in head_preds.items()},
                                       {h: np.concatenate(v) for h, v in head_gold.items()},
                                       slot_preds, slot_gold))
        logger.info("***** Eval results (%s) *****", mode)
        for k in sorted(results):
            logger.info("  %s = %.4f", k, results[k])
        return results

    def save_model(self, out_dir: str):
        self.model.save(out_dir)
        with open(os.path.join(out_dir, "training_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(self.args), f, indent=1, ensure_ascii=False, default=str)
        logger.info("Saving model checkpoint to %s", out_dir)

    def load_model(self):
        if not os.path.exists(self.args.model_out):
            raise FileNotFoundError(f"No existe {self.args.model_out}: entrena primero")
        self.model = JointBERT.load(self.args.model_out, device=self.device)
        logger.info("***** Model loaded from %s *****", self.args.model_out)
