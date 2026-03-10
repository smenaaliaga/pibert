"""
Procesa un archivo Excel y genera los archivos para la arquitectura multi-head del JointBERT:
- data/{dataset}/processed/label (cabezas separadas por comas)
- data/{dataset}/processed/seq.in (texto)
- data/{dataset}/{head}_label.txt (vocabulario por cabeza)
- data/{dataset}/slot_label.txt (etiquetas NER extraidas de NER_BIO)

Estructura esperada del Excel (columnas por nombre):
- Formato legacy: Utterance, CalcMode, ActivityCls, RegionCls, InvestmentCls, ReqForm, NER_BIO
- Formato nuevo (snake_case): utterance, calc_mode, activity, region, investment, req_form, ner_bio

Notas de compatibilidad:
- El script acepta variantes frecuentes como calc_mode_cls, activity_cls, region_cls,
  investment_cls y req_form_cls.
- NER_BIO/ner_bio es opcional. Si no existe, se generan etiquetas de slot "O".

Uso:
    python util/process_raw_data.py --dataset pibimacecv5 --input_file data/pibimacecv5/raw/dataset.xlsx

Notas:
- El archivo Excel debe estar en formato .xlsx o .xls
- Las columnas se buscan por nombre en el header del Excel
- Se generan automaticamente los vocabularios (*_label.txt)
- slot_label.txt se genera desde las etiquetas únicas en NER_BIO
"""

import argparse
import os
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def process_excel_pandas(input_file, sheet_name=None):
    """Lee Excel usando pandas."""
    if not HAS_PANDAS:
        raise ImportError("pandas no está instalado. Instala con: pip install pandas")
    
    # Si sheet_name es None, leer la primera hoja (índice 0)
    if sheet_name is None:
        df = pd.read_excel(input_file, sheet_name=0)
    else:
        df = pd.read_excel(input_file, sheet_name=sheet_name)
    
    return df


def process_file(dataset, input_file, sheet_name=None):
    """Procesa el archivo Excel y genera los archivos de datos."""
    
    base_path = Path('data') / dataset
    processed_path = base_path / 'processed'
    processed_path.mkdir(parents=True, exist_ok=True)

    # Leer Excel con pandas (obligatorio para lectura por nombres de columnas)
    if not HAS_PANDAS:
        raise ImportError("pandas es requerido para procesar el Excel. Instala con: pip install pandas openpyxl")
    
    df = process_excel_pandas(input_file, sheet_name=sheet_name)
    
    # Resolver columnas (compatibilidad entre formato legacy y snake_case)
    def resolve_column(aliases):
        # Match exacto primero
        for name in aliases:
            if name in df.columns:
                return name
        # Match case-insensitive después
        lower_to_original = {str(col).lower(): col for col in df.columns}
        for name in aliases:
            key = str(name).lower()
            if key in lower_to_original:
                return lower_to_original[key]
        return None

    required_column_aliases = {
        'utterance': ['Utterance', 'utterance'],
        'calc_mode': ['CalcMode', 'calc_mode', 'calc_mode_cls'],
        'activity': ['ActivityCls', 'activity', 'activity_cls'],
        'region': ['RegionCls', 'region', 'region_cls'],
        'investment': ['InvestmentCls', 'investment', 'investment_cls'],
        'req_form': ['ReqForm', 'req_form', 'req_form_cls'],
    }
    optional_column_aliases = {
        'ner_bio': ['NER_BIO', 'ner_bio'],
    }

    resolved_columns = {
        key: resolve_column(aliases) for key, aliases in required_column_aliases.items()
    }
    resolved_optional = {
        key: resolve_column(aliases) for key, aliases in optional_column_aliases.items()
    }

    missing_columns = [key for key, col_name in resolved_columns.items() if col_name is None]

    if missing_columns:
        print(f"Error: Las siguientes columnas requeridas no se encuentran en el Excel:")
        print(f"  {', '.join(missing_columns)}")
        print(f"\nColumnas disponibles: {', '.join(df.columns)}")
        raise ValueError(f"Columnas faltantes: {missing_columns}")

    print("Columnas detectadas:")
    for key, col_name in resolved_columns.items():
        print(f"  {key}: {col_name}")
    if resolved_optional['ner_bio']:
        print(f"  ner_bio: {resolved_optional['ner_bio']}")
    else:
        print("  ner_bio: no encontrado (se usarán etiquetas 'O')")
    
    # Nombres canónicos de las cabezas (orden en 'label')
    head_names = ['calc_mode', 'activity', 'region', 'investment', 'req_form']
    
    # Almacenar datos y vocabularios
    seq_in = []
    seq_out = []
    labels = []
    
    # Vocabularios por cabeza (usar dict para preservar orden de aparición)
    vocabularies = {head: {} for head in head_names}
    
    # Vocabulario para slots (etiquetas NER_BIO)
    slot_vocabulary = {}

    # Procesar cada fila del DataFrame
    for idx, row in df.iterrows():
        # Extraer texto
        utterance_col = resolved_columns['utterance']
        text = str(row[utterance_col]).strip() if pd.notna(row[utterance_col]) else ""
        
        # Si text está vacío, saltar
        if not text:
            print(f"Advertencia: Fila {idx+2} tiene texto vacío. Saltando.")
            continue
        
        seq_in.append(text)
        
        # Extraer etiquetas NER
        ner_col = resolved_optional['ner_bio']
        if ner_col is not None and pd.notna(row[ner_col]):
            ner_bio = str(row[ner_col]).strip()
        else:
            ner_bio = ""
        
        # Si no hay etiquetas NER, usar "O" para cada token
        tokens = text.split()
        if not ner_bio or ner_bio.lower() == 'nan':
            slot_tags = " ".join(["O"] * len(tokens))
        else:
            slot_tags = ner_bio
            # Agregar etiquetas al vocabulario de slots
            for tag in ner_bio.split():
                if tag not in slot_vocabulary:
                    slot_vocabulary[tag] = len(slot_vocabulary)
        
        # Verificar que el número de tokens coincida con el número de etiquetas
        if len(slot_tags.split()) != len(tokens):
            print(f"Advertencia: Fila {idx+2} - número de tokens ({len(tokens)}) no coincide con número de etiquetas NER ({len(slot_tags.split())})")
            print(f"  Texto: {text}")
            print(f"  NER: {slot_tags}")
        
        seq_out.append(slot_tags)
        
        # Extraer cabezas clasificadoras
        head_values = []
        for head_name in head_names:
            source_col = resolved_columns[head_name]
            val = str(row[source_col]).strip() if pd.notna(row[source_col]) else ""
            head_values.append(val)
            
            # Agregar al vocabulario si no está vacío
            if val and val.lower() != 'nan':
                if val not in vocabularies[head_name]:
                    vocabularies[head_name][val] = len(vocabularies[head_name])
        
        # Crear línea de label (cabezas separadas por comas)
        label_line = ",".join(head_values)
        labels.append(label_line)
    
    # Escribir archivos procesados
    with open(processed_path / 'seq.in', 'w', encoding='utf-8') as f:
        for line in seq_in:
            f.write(f"{line}\n")

    with open(processed_path / 'label', 'w', encoding='utf-8') as f:
        for line in labels:
            f.write(f"{line}\n")

    with open(processed_path / 'seq.out', 'w', encoding='utf-8') as f:
        for line in seq_out:
            f.write(f"{line}\n")

    # Escribir vocabularios por cabeza
    label_file_map = {
        'calc_mode': 'calc_mode_label.txt',
        'activity': 'activity_label.txt',
        'region': 'region_label.txt',
        'investment': 'investment_label.txt',
        'req_form': 'req_form_label.txt',
    }

    for head_name, vocab in vocabularies.items():
        vocab_file = base_path / label_file_map[head_name]
        # Ordenar por frecuencia (orden de aparición en el vocabulario)
        sorted_labels = sorted(vocab.items(), key=lambda x: x[1])
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for label, _ in sorted_labels:
                f.write(f"{label}\n")
        print(f"Salida: {vocab_file} ({len(sorted_labels)} etiquetas)")

    # Escribir slot_label.txt con etiquetas NER únicas
    if not slot_vocabulary:
        # Si no hay etiquetas NER en los datos, crear solo con "O"
        slot_vocabulary = {"O": 0}
    elif "O" not in slot_vocabulary:
        # Asegurar que "O" está en el vocabulario
        slot_vocabulary["O"] = len(slot_vocabulary)
    
    slot_vocab_file = base_path / 'slot_label.txt'
    # Ordenar etiquetas (O primero, luego el resto alfabéticamente)
    sorted_slot_labels = sorted(slot_vocabulary.items(), key=lambda x: (x[0] != "O", x[0]))
    with open(slot_vocab_file, 'w', encoding='utf-8') as f:
        for label, _ in sorted_slot_labels:
            f.write(f"{label}\n")
    print(f"Salida: {slot_vocab_file} ({len(sorted_slot_labels)} etiquetas)")

    # Resumen
    print(f"\nProcesadas {len(seq_in)} filas")
    print(f"Salida: {processed_path / 'seq.in'}")
    print(f"Salida: {processed_path / 'seq.out'}")
    print(f"Salida: {processed_path / 'label'}")
    print("\nVocabularios generados:")
    for head_name in head_names:
        print(f"  {head_name}: {len(vocabularies[head_name])} etiquetas")
    print(f"  Slots (NER): {len(slot_vocabulary)} etiquetas")


def main():
    parser = argparse.ArgumentParser(
        description='Procesa un Excel con arquitectura multi-head para JointBERT. '
                    'Acepta columnas legacy (Utterance, CalcMode, ActivityCls, RegionCls, '
                    'InvestmentCls, ReqForm, NER_BIO) y snake_case '
                    '(utterance, calc_mode, activity, region, investment, req_form, ner_bio)'
    )
    parser.add_argument('--dataset', type=str, required=True, help='Nombre del dataset (carpeta en data/)')
    parser.add_argument('--input_file', type=str, required=True, help='Ruta al archivo Excel')
    parser.add_argument('--sheet_name', type=str, default=None, help='Nombre de la hoja del Excel (por defecto: primera hoja)')
    args = parser.parse_args()

    process_file(args.dataset, args.input_file, sheet_name=args.sheet_name)


if __name__ == '__main__':
    main()
