"""
Procesa un archivo CSV raw (estructura como el ejemplo `project-1-at-...csv`) y genera los archivos:
- data/{dataset}/processed/label
- data/{dataset}/processed/seq.in
- data/{dataset}/processed/seq.out
- data/{dataset}/intent_label.txt
- data/{dataset}/slot_label.txt

Uso:
    python util/process_raw_labels.py --dataset pibimacec --input_file data/pibimacec/raw/project-1-...csv

Notas:
- `label` se extrae desde la columna `calc_mode`.
- `seq.in` se extrae desde la columna `text`.
- `seq.out` se construye a partir de las anotaciones en la columna `ner` (lista de spans con start/end y labels).

El script intenta parsear la columna `ner` como JSON; si falla intenta arreglar dobles comillas o usar ast.literal_eval.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import ast


def parse_ner_field(ner_raw):
    """Intenta convertir la cadena ner a una lista de dicts con keys start,end,labels."""
    if ner_raw is None:
        return []
    ner_raw = ner_raw.strip()
    if ner_raw == "":
        return []
    # Try json loads directly
    try:
        return json.loads(ner_raw)
    except Exception:
        pass
    # Try to fix doubled double-quotes that appear in some exports
    try:
        fixed = ner_raw.replace('""', '"')
        return json.loads(fixed)
    except Exception:
        pass
    # Try ast literal eval
    try:
        return ast.literal_eval(ner_raw)
    except Exception:
        pass
    # Last resort: return empty
    return []


def text_to_tokens_with_offsets(text):
    """Tokeniza por espacios y devuelve tokens con offsets (start,end)."""
    tokens = []
    idx = 0
    for token in text.split():
        # find token in text starting from idx
        start = text.find(token, idx)
        if start == -1:
            # fallback: approximate
            start = idx
        end = start + len(token)
        tokens.append((token, start, end))
        idx = end
    return tokens


def build_bio_for_text(text, annotations):
    """
    Dado un `text` y `annotations` (lista de dicts con start,end,labels), construye una secuencia BIO por token.
    Las `labels` en las anotaciones pueden ser un array; se toma la primera.
    Se normaliza la etiqueta (por ejemplo INDICATOR -> indicator) para formar B-indicator / I-indicator.
    """
    tokens = text_to_tokens_with_offsets(text)
    bio_tags = ["O"] * len(tokens)

    for ann in annotations:
        # ann expected keys: start, end, labels
        try:
            astart = int(ann.get('start'))
            aend = int(ann.get('end'))
            labs = ann.get('labels') or ann.get('label') or []
            if isinstance(labs, list) and len(labs) > 0:
                lab = labs[0]
            elif isinstance(labs, str):
                lab = labs
            else:
                lab = None
            if lab is None:
                continue
            # normalize
            lab_norm = str(lab).strip().lower()
            lab_norm = lab_norm.replace(' ', '_')
        except Exception:
            continue

        # assign tags to tokens whose span intersects [astart, aend)
        first = True
        for i, (tok, tstart, tend) in enumerate(tokens):
            # overlap check
            if tstart < aend and tend > astart:
                prefix = 'B-' if first else 'I-'
                bio_tags[i] = prefix + lab_norm
                first = False
    return [t for t,_,_ in tokens], bio_tags


def process_file(dataset, input_file):
    base_path = Path('data') / dataset
    raw_path = Path(input_file)
    processed_path = base_path / 'processed'
    processed_path.mkdir(parents=True, exist_ok=True)

    labels = []       # calc_mode per row -> label file
    seq_in = []       # text per row -> seq.in
    seq_out = []      # BIO per row -> seq.out

    all_intents = set()
    all_slot_labels = set()

    # read CSV (robust to comma or tab separators)
    # Use utf-8-sig to gracefully handle BOM if present
    with open(raw_path, 'r', encoding='utf-8-sig') as csvfile:
        # Read a sample to guess the delimiter (comma, tab or semicolon)
        sample = csvfile.read(4096)
        csvfile.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[',', '\t', ';'])
            delimiter = dialect.delimiter
        except Exception:
            # fallback to comma
            delimiter = ','

        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            # calc_mode -> label
            calc_mode = row.get('calc_mode') or row.get('intencion') or row.get('intencion')
            if calc_mode is None:
                calc_mode = ''
            calc_mode = calc_mode.strip()
            labels.append(calc_mode)
            if calc_mode:
                all_intents.add(calc_mode)

            # text -> seq.in
            text = row.get('text') or ''
            text = text.strip()
            seq_in.append(text)

            # ner -> annotations
            ner_raw = row.get('ner') or row.get('ner_annotations') or row.get('annotation')
            annotations = parse_ner_field(ner_raw)

            # Build BIO tags
            tokens, tags = build_bio_for_text(text, annotations)

            # Normalize tags: prefix B-/I- with lowercase label names (as in slot_label.txt)
            normalized_tags = []
            for tag in tags:
                if tag == 'O':
                    normalized_tags.append('O')
                else:
                    # ensure format like B-period or I-indicator
                    normalized_tags.append(tag)
                    all_slot_labels.add(tag)

            # join tokens and tags per token, but seq.out usually is sequence of tags per token
            seq_out.append(' '.join(normalized_tags))

    # write processed files
    with open(processed_path / 'label', 'w', encoding='utf-8') as f:
        for line in labels:
            f.write(f"{line}\n")

    with open(processed_path / 'seq.in', 'w', encoding='utf-8') as f:
        for line in seq_in:
            f.write(f"{line}\n")

    with open(processed_path / 'seq.out', 'w', encoding='utf-8') as f:
        for line in seq_out:
            f.write(f"{line}\n")

    # intent_label.txt in base_path
    intent_labels = sorted(all_intents)
    with open(base_path / 'intent_label.txt', 'w', encoding='utf-8') as f:
        for lab in intent_labels:
            f.write(f"{lab}\n")

    # slot_label.txt: generate B- and I- variants from discovered labels
    # all_slot_labels contains strings like 'B-period' or 'I-indicator'
    # extract base labels
    base_labels = set()
    for t in all_slot_labels:
        if t == 'O':
            continue
        if t.startswith('B-') or t.startswith('I-'):
            base_labels.add(t[2:])
        else:
            # in case tags are like 'period' treat as base
            base_labels.add(t)

    slot_ordered = []
    for b in sorted(base_labels):
        slot_ordered.append('B-' + b)
    for b in sorted(base_labels):
        slot_ordered.append('I-' + b)
    slot_ordered.append('O')

    with open(base_path / 'slot_label.txt', 'w', encoding='utf-8') as f:
        for s in slot_ordered:
            f.write(f"{s}\n")

    print(f"Processed {len(labels)} rows")
    print(f"Wrote: {processed_path / 'label'}")
    print(f"Wrote: {processed_path / 'seq.in'}")
    print(f"Wrote: {processed_path / 'seq.out'}")
    print(f"Wrote: {base_path / 'intent_label.txt'}")
    print(f"Wrote: {base_path / 'slot_label.txt'}")


def main():
    parser = argparse.ArgumentParser(description='Process raw csv annotations to processed format')
    parser.add_argument('--dataset', type=str, required=True, help='dataset folder name under data/')
    parser.add_argument('--input_file', type=str, required=True, help='path to raw csv file')
    args = parser.parse_args()

    process_file(args.dataset, args.input_file)


if __name__ == '__main__':
    main()
