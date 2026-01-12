import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import BertConfig #, DistilBertConfig, AlbertConfig
from transformers import BertTokenizer #, DistilBertTokenizer, AlbertTokenizer

from model_in import JointBERT #, JointDistilBERT, JointAlbert

MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
    'beto': (BertConfig, JointBERT, BertTokenizer),  # BETO usa la misma arquitectura que BERT
    # 'distilbert': (DistilBertConfig, JointDistilBERT, DistilBertTokenizer),
    # 'albert': (AlbertConfig, JointAlbert, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'beto': 'dccuchile/bert-base-spanish-wwm-cased',  # BETO del Centro de Modelamiento Matemático, U. de Chile
    # 'distilbert': 'distilbert-base-uncased',
    # 'albert': 'albert-xxlarge-v1'
}


def get_indicator_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.indicator_label_file), 'r', encoding='utf-8')]

def get_metric_type_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.metric_type_label_file), 'r', encoding='utf-8')]

def get_calc_mode_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.calc_mode_label_file), 'r', encoding='utf-8')]

def get_seasonal_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.seasonal_label_file), 'r', encoding='utf-8')]

def get_req_form_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.req_form_label_file), 'r', encoding='utf-8')]

def get_frequency_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.frequency_label_file), 'r', encoding='utf-8')]

def get_activity_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.activity_label_file), 'r', encoding='utf-8')]

# def get_slot_labels(args):
#     return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


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


def compute_metrics(indicator_preds, indicator_labels,
                    metric_type_preds, metric_type_labels,
                    calc_mode_preds, calc_mode_labels,
                    seasonal_preds, seasonal_labels,
                    req_form_preds, req_form_labels,
                    frequency_preds, frequency_labels,
                    activity_preds, activity_labels,
                    slot_preds, slot_labels):
    assert len(indicator_preds) == len(indicator_labels) == len(metric_type_preds) == len(metric_type_labels)
    assert len(calc_mode_preds) == len(calc_mode_labels) == len(seasonal_preds) == len(seasonal_labels)
    assert len(req_form_preds) == len(req_form_labels) == len(frequency_preds) == len(frequency_labels)
    assert len(activity_preds) == len(activity_labels) == len(slot_preds) == len(slot_labels)

    results = {}
    results.update(get_indicator_acc(indicator_preds, indicator_labels))
    results.update(get_metric_type_acc(metric_type_preds, metric_type_labels))
    results.update(get_calc_mode_acc(calc_mode_preds, calc_mode_labels))
    results.update(get_seasonal_acc(seasonal_preds, seasonal_labels))
    results.update(get_req_form_acc(req_form_preds, req_form_labels))
    results.update(get_frequency_acc(frequency_preds, frequency_labels))
    results.update(get_activity_acc(activity_preds, activity_labels))
    results.update(get_slot_metrics(slot_preds, slot_labels))

    semantic_result = get_sentence_frame_acc(indicator_preds, indicator_labels,
                                             metric_type_preds, metric_type_labels,
                                             calc_mode_preds, calc_mode_labels,
                                             seasonal_preds, seasonal_labels,
                                             req_form_preds, req_form_labels,
                                             frequency_preds, frequency_labels,
                                             activity_preds, activity_labels,
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


def get_indicator_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "indicator_acc": acc
    }


def get_metric_type_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "metric_type_acc": acc
    }


def get_calc_mode_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "calc_mode_acc": acc
    }


def get_seasonal_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "seasonal_acc": acc
    }


def get_req_form_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "req_form_acc": acc
    }


def get_frequency_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "frequency_acc": acc
    }


def get_activity_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "activity_acc": acc
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(indicator_preds, indicator_labels,
                           metric_type_preds, metric_type_labels,
                           calc_mode_preds, calc_mode_labels,
                           seasonal_preds, seasonal_labels,
                           req_form_preds, req_form_labels,
                           frequency_preds, frequency_labels,
                           activity_preds, activity_labels,
                           slot_preds, slot_labels):
    """Semantic frame accuracy: todas las cabezas y todos los slots correctos en la oración."""
    indicator_result = (indicator_preds == indicator_labels)
    metric_type_result = (metric_type_preds == metric_type_labels)
    calc_mode_result = (calc_mode_preds == calc_mode_labels)
    seasonal_result = (seasonal_preds == seasonal_labels)
    req_form_result = (req_form_preds == req_form_labels)
    frequency_result = (frequency_preds == frequency_labels)
    activity_result = (activity_preds == activity_labels)

    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = all(p == l for p, l in zip(preds, labels))
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    semantic_acc = (indicator_result & metric_type_result & calc_mode_result &
                    seasonal_result & req_form_result & frequency_result & activity_result & slot_result).mean()
    return {
        "semantic_frame_acc": semantic_acc
    }