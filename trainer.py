import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from utils import MODEL_CONFIG, compute_metrics, get_calc_mode_labels, get_activity_labels, get_region_labels, get_investment_labels, get_req_form_labels, get_slot_labels

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.calc_mode_label_lst = get_calc_mode_labels(args)
        self.activity_label_lst = get_activity_labels(args)
        self.region_label_lst = get_region_labels(args)
        self.investment_label_lst = get_investment_labels(args)
        self.req_form_label_lst = get_req_form_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = MODEL_CONFIG
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      # heads
                                                      calc_mode_label_lst=self.calc_mode_label_lst,
                                                      activity_label_lst=self.activity_label_lst,
                                                      region_label_lst=self.region_label_lst,
                                                      investment_label_lst=self.investment_label_lst,
                                                      req_form_label_lst=self.req_form_label_lst,
                                                      slot_label_lst=self.slot_label_lst)

        # GPU or CPU or MPS (Apple Silicon)
        if torch.cuda.is_available() and not args.no_cuda:
            self.device = "cuda"
        elif torch.backends.mps.is_available() and not args.no_cuda:
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        best_val_loss = float("inf")
        epochs_no_improve = 0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for epoch_idx, _ in enumerate(train_iterator):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'calc_mode_label_ids': batch[3],
                    'activity_label_ids': batch[4],
                    'region_label_ids': batch[5],
                    'investment_label_ids': batch[6],
                    'req_form_label_ids': batch[7],
                }
                if self.args.slot_loss_coef != 0:
                    inputs['slot_labels_ids'] = batch[8]
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate("dev")

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

            if self.args.early_stopping:
                val_results = self.evaluate("dev")
                val_loss = val_results.get("loss", float("inf"))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    self.save_model()
                    logger.info("Early stopping: improved val loss to %.4f at epoch %d", best_val_loss, epoch_idx + 1)
                else:
                    epochs_no_improve += 1
                    logger.info("Early stopping: no improvement for %d epoch(s)", epochs_no_improve)

                if epochs_no_improve >= self.args.early_stopping_patience:
                    logger.info("Early stopping triggered after %d epochs without improvement", epochs_no_improve)
                    break

        return global_step, tr_loss / max(global_step, 1)

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        
        eval_loss = 0.0
        nb_eval_steps = 0
        calc_mode_preds = None
        activity_preds = None
        region_preds = None
        investment_preds = None
        req_form_preds = None
        slot_preds = None
        
        out_calc_mode_label_ids = None
        out_activity_label_ids = None
        out_region_label_ids = None
        out_investment_label_ids = None
        out_req_form_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'calc_mode_label_ids': batch[3],
                          'activity_label_ids': batch[4],
                          'region_label_ids': batch[5],
                          'investment_label_ids': batch[6],
                          'req_form_label_ids': batch[7],
                          }
                if self.args.slot_loss_coef != 0:
                    inputs['slot_labels_ids'] = batch[8]
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss, (calc_mode_logits, activity_logits, region_logits, investment_logits, req_form_logits, slot_logits) = outputs[:2] 

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Calc Mode prediction
            if calc_mode_preds is None:
                calc_mode_preds = calc_mode_logits.detach().cpu().numpy()
                out_calc_mode_label_ids = inputs['calc_mode_label_ids'].detach().cpu().numpy()
            else:
                calc_mode_preds = np.append(calc_mode_preds, calc_mode_logits.detach().cpu().numpy(), axis=0)
                out_calc_mode_label_ids = np.append(
                    out_calc_mode_label_ids, inputs['calc_mode_label_ids'].detach().cpu().numpy(), axis=0)
                
            # Activity prediction
            if activity_preds is None:
                activity_preds = activity_logits.detach().cpu().numpy()
                out_activity_label_ids = inputs['activity_label_ids'].detach().cpu().numpy()
            else:
                activity_preds = np.append(activity_preds, activity_logits.detach().cpu().numpy(), axis=0)
                out_activity_label_ids = np.append(
                    out_activity_label_ids, inputs['activity_label_ids'].detach().cpu().numpy(), axis=0)
                
            # Region prediction
            if region_preds is None:
                region_preds = region_logits.detach().cpu().numpy()
                out_region_label_ids = inputs['region_label_ids'].detach().cpu().numpy()
            else:
                region_preds = np.append(region_preds, region_logits.detach().cpu().numpy(), axis=0)
                out_region_label_ids = np.append(
                    out_region_label_ids, inputs['region_label_ids'].detach().cpu().numpy(), axis=0)

            # Investment prediction
            if investment_preds is None:
                investment_preds = investment_logits.detach().cpu().numpy()
                out_investment_label_ids = inputs['investment_label_ids'].detach().cpu().numpy()
            else:
                investment_preds = np.append(investment_preds, investment_logits.detach().cpu().numpy(), axis=0)
                out_investment_label_ids = np.append(
                    out_investment_label_ids, inputs['investment_label_ids'].detach().cpu().numpy(), axis=0)

            # Req Form prediction
            if req_form_preds is None:
                req_form_preds = req_form_logits.detach().cpu().numpy()
                out_req_form_label_ids = inputs['req_form_label_ids'].detach().cpu().numpy()
            else:
                req_form_preds = np.append(req_form_preds, req_form_logits.detach().cpu().numpy(), axis=0)
                out_req_form_label_ids = np.append(
                    out_req_form_label_ids, inputs['req_form_label_ids'].detach().cpu().numpy(), axis=0)

            # Slot prediction
            if self.args.slot_loss_coef != 0:
                if slot_preds is None:
                    if self.args.use_crf:
                        if hasattr(self.model.crf, 'decode'):
                            decoded = self.model.crf.decode(slot_logits, mask=inputs['attention_mask'].bool())
                        else:
                            decoded = self.model.crf.viterbi_decode(slot_logits, inputs['attention_mask'].bool())
                        if isinstance(decoded, torch.Tensor):
                            decoded = decoded.detach().cpu().tolist()
                        elif isinstance(decoded, np.ndarray):
                            decoded = decoded.tolist()
                        else:
                            decoded = [list(seq) for seq in decoded]
                        slot_preds = decoded
                    else:
                        slot_preds = slot_logits.detach().cpu().numpy()

                    out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
                else:
                    if self.args.use_crf:
                        if hasattr(self.model.crf, 'decode'):
                            decoded = self.model.crf.decode(slot_logits, mask=inputs['attention_mask'].bool())
                        else:
                            decoded = self.model.crf.viterbi_decode(slot_logits, inputs['attention_mask'].bool())
                        if isinstance(decoded, torch.Tensor):
                            decoded = decoded.detach().cpu().tolist()
                        elif isinstance(decoded, np.ndarray):
                            decoded = decoded.tolist()
                        else:
                            decoded = [list(seq) for seq in decoded]
                        slot_preds.extend(decoded)
                    else:
                        slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                    out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Heads result
        calc_mode_preds = np.argmax(calc_mode_preds, axis=1)
        activity_preds = np.argmax(activity_preds, axis=1)
        region_preds = np.argmax(region_preds, axis=1)
        investment_preds = np.argmax(investment_preds, axis=1)
        req_form_preds = np.argmax(req_form_preds, axis=1)
    
        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = compute_metrics(calc_mode_preds, out_calc_mode_label_ids, 
                                       activity_preds, out_activity_label_ids, 
                                       region_preds, out_region_label_ids, 
                                       investment_preds, out_investment_label_ids,
                                       req_form_preds, out_req_form_label_ids, 
                                       slot_preds_list, out_slot_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_out):
            os.makedirs(self.args.model_out)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_out)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_out, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_out)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_out):
            raise Exception("Model doesn't exists! Train first!")

        try:
            model_config = self.config_class.from_pretrained(self.args.model_out)
            self.model = self.model_class.from_pretrained(self.args.model_out,
                                                          config=model_config,
                                                          args=self.args,
                                                          calc_mode_label_lst=self.calc_mode_label_lst,
                                                          activity_label_lst=self.activity_label_lst,
                                                          region_label_lst=self.region_label_lst,
                                                          investment_label_lst=self.investment_label_lst,
                                                          req_form_label_lst=self.req_form_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except Exception as e:
            raise Exception(f"Error loading model from {self.args.model_out}: {e}")