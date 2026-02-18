import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import AutoConfig
from transformers import AutoTokenizer  # usar tokenizer rápido para generar tokenizer.json

from model_in import JointBERT

MODEL_CONFIG = (AutoConfig, JointBERT, AutoTokenizer)

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'beto': 'dccuchile/bert-base-spanish-wwm-cased',
    'deberta': 'microsoft/mdeberta-v3-base',
    'roberta': 'bertin-project/bertin-roberta-base-spanish',
}

def get_calc_mode_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.calc_mode_label_file), 'r', encoding='utf-8')]

def get_activity_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.activity_label_file), 'r', encoding='utf-8')]

def get_region_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.region_label_file), 'r', encoding='utf-8')]

def get_investment_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.investment_label_file), 'r', encoding='utf-8')]

def get_req_form_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.req_form_label_file), 'r', encoding='utf-8')]

def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    elif not args.no_cuda and torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)


def compute_metrics(calc_mode_preds, calc_mode_labels,
                    activity_preds, activity_labels,
                    region_preds, region_labels,
                    investment_preds, investment_labels,
                    req_form_preds, req_form_labels,
                    slot_preds, slot_labels):
    assert len(calc_mode_preds) == len(calc_mode_labels) \
        == len(activity_preds) == len(activity_labels) == len(region_preds) == len(region_labels) \
        == len(investment_preds) == len(investment_labels) \
        == len(req_form_preds) == len(req_form_labels) == len(slot_preds) == len(slot_labels)
        

    results = {}
    results.update(get_calc_mode_acc(calc_mode_preds, calc_mode_labels))
    results.update(get_activity_acc(activity_preds, activity_labels))
    results.update(get_region_acc(region_preds, region_labels))
    results.update(get_investment_acc(investment_preds, investment_labels))
    results.update(get_req_form_acc(req_form_preds, req_form_labels))
    results.update(get_slot_metrics(slot_preds, slot_labels))

    semantic_result = get_sentence_frame_acc(calc_mode_preds, calc_mode_labels,
                                             activity_preds, activity_labels,
                                             region_preds, region_labels,
                                             investment_preds, investment_labels,
                                             req_form_preds, req_form_labels,
                                             slot_preds, slot_labels)
    results.update(semantic_result)

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_calc_mode_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "calc_mode_acc": acc
    }


def get_activity_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "activity_acc": acc
    }


def get_region_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "region_acc": acc
    }


def get_investment_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "investment_acc": acc
    }


def get_req_form_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "req_form_acc": acc
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(calc_mode_preds, calc_mode_labels,
                           activity_preds, activity_labels,
                           region_preds, region_labels,
                           investment_preds, investment_labels,
                           req_form_preds, req_form_labels,
                           slot_preds, slot_labels):
    """Semantic frame accuracy: todas las cabezas y todos los slots correctos en la oración."""
    calc_mode_result = (calc_mode_preds == calc_mode_labels)
    req_form_result = (req_form_preds == req_form_labels)
    activity_result = (activity_preds == activity_labels)
    region_result = (region_preds == region_labels)
    investment_result = (investment_preds == investment_labels)

    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = all(p == l for p, l in zip(preds, labels))
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    semantic_acc = (calc_mode_result & activity_result & region_result & investment_result & req_form_result & slot_result).mean()
    
    return {
        "semantic_frame_acc": semantic_acc
    }