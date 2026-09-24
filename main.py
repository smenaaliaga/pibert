import argparse
import json
import os

from data_loader import load_and_cache_examples
from schema import Schema
from trainer import Trainer
from utils import MODEL_PATH_MAP, init_logger, load_tokenizer, pick_device, resolve_model_name, set_seed


def main(args):
    init_logger()
    set_seed(args.seed, pick_device(args.no_cuda))
    schema = Schema.load(args.schema)
    tokenizer = load_tokenizer(args.model_name_or_path)

    datasets = {m: load_and_cache_examples(args, tokenizer, schema, m) for m in ("train", "dev", "test")}
    trainer = Trainer(args, schema, datasets["train"], datasets["dev"], datasets["test"])

    if args.do_train:
        trainer.train()
        tokenizer.save_pretrained(args.model_out)

    if args.do_eval:
        trainer.load_model()
        results = trainer.evaluate("test")
        with open(os.path.join(args.model_out, "eval_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="Nombre del dataset en data/")
    p.add_argument("--model_out", required=True, help="Nombre del modelo (se guarda en model_out/)")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--schema", default=None, help="schema.json (default: data/<task>/schema.json)")
    p.add_argument("--model_type", default="beto",
                   help="Atajo (" + ", ".join(MODEL_PATH_MAP) + ") o identificador de Hugging Face")

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--max_seq_len", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_epochs", type=float, default=10.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--dropout_rate", type=float, default=0.1)
    p.add_argument("--early_stopping", action="store_true")
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_eval", action="store_true")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--ignore_index", type=int, default=-100)
    p.add_argument("--slot_loss_coef", type=float, default=1.0)
    p.add_argument("--use_crf", action="store_true")

    args = p.parse_args()
    args.model_name_or_path = resolve_model_name(args.model_type)
    args.model_out = os.path.join("model_out", args.model_out)
    args.schema = args.schema or os.path.join(args.data_dir, args.task, "schema.json")
    main(args)
