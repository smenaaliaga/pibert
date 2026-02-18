import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

from utils import (
    get_calc_mode_labels,
    get_activity_labels,
    get_region_labels,
    get_investment_labels,
    get_req_form_labels,
    get_slot_labels,
)

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, calc_mode_label=None, activity_label=None, region_label=None, investment_label=None, req_form_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.calc_mode_label = calc_mode_label
        self.activity_label = activity_label
        self.region_label = region_label
        self.investment_label = investment_label
        self.req_form_label = req_form_label
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, 
                 calc_mode_label_id, activity_label_id, 
                 investment_label_id,
                 region_label_id, req_form_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.calc_mode_label_id = calc_mode_label_id
        self.activity_label_id = activity_label_id
        self.investment_label_id = investment_label_id
        self.region_label_id = region_label_id
        self.req_form_label_id = req_form_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.calc_mode_labels = get_calc_mode_labels(args)
        self.activity_labels = get_activity_labels(args)
        self.region_labels = get_region_labels(args)
        self.investment_labels = get_investment_labels(args)
        self.req_form_labels = get_req_form_labels(args)
        self.slot_labels = get_slot_labels(args)

        self.input_text_file = 'seq.in'
        self.label_file = 'label'
        self.slot_labels_file = 'seq.out'

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, labels, slots, set_type):  # slots parameter removed
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, label, slot) in enumerate(zip(texts, labels, slots)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. heads from label line
            # Support: either a single label or comma-separated labels for multiple heads
            calc_mode_label = 0
            activity_label = 0
            region_label = 0
            investment_label = 0
            req_form_label = 0
            slot_labels = []
            o_label_idx = self.slot_labels.index("O") if "O" in self.slot_labels else 0
            for s in slot.split():
                slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else o_label_idx)
            assert len(words) == len(slot_labels)

            def map_or_default(name, label_list, default_idx=0):
                if not isinstance(name, str):
                    return default_idx
                name = name.strip()
                if not label_list:
                    return default_idx
                return label_list.index(name) if name in label_list else default_idx

            if "," in label:
                parts = [p.strip() for p in label.split(",")]
                # Formato esperado pibimacecv5: calc_mode,activity,region,investment,req_form
                while len(parts) < 5:
                    parts.append("")
                calc_mode_name, activity_name, region_name, investment_name, req_form_name = parts[:5]
                
                calc_mode_label = map_or_default(calc_mode_name, getattr(self, 'calc_mode_labels', []), 0)
                activity_label = map_or_default(activity_name, getattr(self, 'activity_labels', []), 0)
                region_label = map_or_default(region_name, getattr(self, 'region_labels', []), 0)
                investment_label = map_or_default(investment_name, getattr(self, 'investment_labels', []), 0)
                req_form_label = map_or_default(req_form_name, getattr(self, 'req_form_labels', []), 0)

            examples.append(InputExample(
                guid=guid,
                words=words,
                calc_mode_label=calc_mode_label,
                activity_label=activity_label,
                region_label=region_label,
                investment_label=investment_label,
                req_form_label=req_form_label,
                slot_labels=slot_labels
            ))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     labels=self._read_file(os.path.join(data_path, self.label_file)),
                                     slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
                                     set_type=mode)


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("calc_mode_label: %d" % example.calc_mode_label)
            logger.info("activity_label: %d" % example.activity_label)
            logger.info("region_label: %d" % example.region_label)
            logger.info("investment_label: %d" % example.investment_label)
            logger.info("req_form_label: %d" % example.req_form_label)
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          calc_mode_label_id=example.calc_mode_label,
                          activity_label_id=example.activity_label,
                          investment_label_id=example.investment_label,
                          region_label_id=example.region_label,
                          req_form_label_id=example.req_form_label,
                          slot_labels_ids=slot_labels_ids
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    # Use JointProcessor for all datasets
    processor = JointProcessor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}_v2'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file, weights_only=False)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_calc_mode_label_ids = torch.tensor([getattr(f, 'calc_mode_label_id', 0) for f in features], dtype=torch.long)
    all_activity_label_ids = torch.tensor([getattr(f, 'activity_label_id', 0) for f in features], dtype=torch.long)
    all_region_label_ids = torch.tensor([getattr(f, 'region_label_id', 0) for f in features], dtype=torch.long)
    all_investment_label_ids = torch.tensor([getattr(f, 'investment_label_id', 0) for f in features], dtype=torch.long)
    all_req_form_label_ids = torch.tensor([getattr(f, 'req_form_label_id', 0) for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_calc_mode_label_ids,
        all_activity_label_ids,
        all_region_label_ids,
        all_investment_label_ids,
        all_req_form_label_ids,
        all_slot_labels_ids,
    )
    return dataset