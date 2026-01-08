"""
Procesa un archivo Excel y genera los archivos para la arquitectura multi-head del JointBERT:
- data/{dataset}/processed/label (cabezas separadas por comas)
- data/{dataset}/processed/seq.in (texto)
- data/{dataset}/{head}_label.txt (vocabulario por cabeza)
- data/{dataset}/slot_label.txt (vacío, solo contiene "O")

Estructura esperada del Excel:
- Columna 1: text (entrada de texto)
- Columna 2: intent
- Columna 3: indicator
- Columna 4: metric_type
- Columna 5: calc_mode
- Columna 6: seasonal
- Columna 7: req_form
- Columna 8: context_mode
- Columna 9: activity

Uso:
    python util/process_excel_multi_heads.py --dataset pibimacec --input_file data/pibimacec/raw/data.xlsx --sheet_name Sheet1
    
    python util/process_excel_multi_heads.py --dataset pibimacecv2 --input_file data/pibimacecv2/raw/data.xlsx --sheet_name final

Notas:
- El archivo Excel debe estar en formato .xlsx o .xls
- Las columnas se leen en orden: text, intent, indicator, metric_type, calc_mode, seasonal, req_form, context_mode, activity
- Se generan automaticamente los vocabularios (*_label.txt)
- slot_label.txt se crea solo con "O" (slots no se usan en esta arquitectura por defecto)
"""

import argparse
import os
from pathlib import Path

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def process_excel_openpyxl(input_file, sheet_name=None):
    """Lee Excel usando openpyxl."""
    if not HAS_OPENPYXL:
        raise ImportError("openpyxl no está instalado. Instala con: pip install openpyxl")
    
    wb = openpyxl.load_workbook(input_file)
    if sheet_name:
        ws = wb[sheet_name]
    else:
        ws = wb.active
    
    rows = []
    for row in ws.iter_rows(values_only=True):
        rows.append(row)
    
    return rows


def process_excel_pandas(input_file, sheet_name=None):
    """Lee Excel usando pandas."""
    if not HAS_PANDAS:
        raise ImportError("pandas no está instalado. Instala con: pip install pandas")
    
    df = pd.read_excel(input_file, sheet_name=sheet_name)
    # Convertir a lista de tuplas (fila por fila)
    rows = [tuple(row) for row in df.values]
    return rows


def process_file(dataset, input_file, sheet_name=None, skip_header=True):
    """Procesa el archivo Excel y genera los archivos de datos."""
    
    base_path = Path('data') / dataset
    processed_path = base_path / 'processed'
    processed_path.mkdir(parents=True, exist_ok=True)

    # Intentar leer con pandas primero (más robusto), luego openpyxl
    try:
        rows = process_excel_pandas(input_file, sheet_name=sheet_name)
    except ImportError:
        rows = process_excel_openpyxl(input_file, sheet_name=sheet_name)
    
    # Headers: text, intent, indicator, metric_type, calc_mode, seasonal, req_form, context_mode, activity
    head_names = ['text', 'intent', 'indicator', 'metric_type', 'calc_mode', 'seasonal', 'req_form', 'context_mode', 'activity']
    
    # Almacenar datos y vocabularios
    seq_in = []
    seq_out = []
    labels = []
    
    # Vocabularios por cabeza (usar dict para preservar orden de aparición)
    vocabularies = {
        'intent': {},
        'indicator': {},
        'metric_type': {},
        'calc_mode': {},
        'seasonal': {},
        'req_form': {},
        'context_mode': {},
        'activity': {}
    }

    # Procesar cada fila del Excel
    for row_idx, row in enumerate(rows):
        # Omitir fila de encabezado si corresponde
        if skip_header and row_idx == 0:
            # Heurística: si la primera celda contiene 'text' o la segunda 'intent', asumimos encabezado
            first_cell = str(row[0]).strip().lower() if len(row) > 0 and row[0] else ""
            second_cell = str(row[1]).strip().lower() if len(row) > 1 and row[1] else ""
            if first_cell == 'text' or second_cell == 'intent':
                continue
        
        # Asegurar que tenemos suficientes columnas
        if len(row) < len(head_names):
            print(f"Advertencia: Fila {row_idx} tiene menos columnas que esperado. Saltando.")
            continue
        
        # Extraer valores
        text = str(row[0]).strip() if row[0] else ""
        
        # Si text está vacío, saltar
        if not text:
            continue
        
        seq_in.append(text)
        
        # Crear seq.out con etiquetas "O" por cada token
        tokens = text.split()
        slot_tags = " ".join(["O"] * len(tokens))
        seq_out.append(slot_tags)
        
        # Extraer cabezas
        head_values = []
        for col_idx in range(1, len(head_names)):
            val = str(row[col_idx]).strip() if row[col_idx] else ""
            head_name = head_names[col_idx]
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
    for head_name, vocab in vocabularies.items():
        vocab_file = base_path / f"{head_name}_label.txt"
        # Ordenar por frecuencia (orden de aparición en el vocabulario)
        sorted_labels = sorted(vocab.items(), key=lambda x: x[1])
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for label, _ in sorted_labels:
                f.write(f"{label}\n")
        print(f"Escribí: {vocab_file} ({len(sorted_labels)} etiquetas)")

    # Escribir slot_label.txt vacío (solo "O")
    with open(base_path / 'slot_label.txt', 'w', encoding='utf-8') as f:
        f.write("O\n")
    print(f"Escribí: {base_path / 'slot_label.txt'} (solo O)")

    # Resumen
    print(f"\nProcesadas {len(seq_in)} filas")
    print(f"Escribí: {processed_path / 'seq.in'}")
    print(f"Escribí: {processed_path / 'seq.out'}")
    print(f"Escribí: {processed_path / 'label'}")
    print("\nVocabularios generados:")
    for head_name in head_names[1:]:
        print(f"  {head_name}: {len(vocabularies[head_name])} etiquetas")


def main():
    parser = argparse.ArgumentParser(
        description='Procesa un Excel con arquitectura multi-head para JointBERT'
    )
    parser.add_argument('--dataset', type=str, required=True, help='Nombre del dataset (carpeta en data/)')
    parser.add_argument('--input_file', type=str, required=True, help='Ruta al archivo Excel')
    parser.add_argument('--sheet_name', type=str, default=None, help='Nombre de la hoja del Excel (por defecto: primera hoja)')
    parser.add_argument('--no_skip_header', action='store_true', help='No saltar la primera fila aunque sea encabezado')
    args = parser.parse_args()

    process_file(args.dataset, args.input_file, sheet_name=args.sheet_name, skip_header=not args.no_skip_header)


if __name__ == '__main__':
    main()
