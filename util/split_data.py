"""
Script para dividir datos en conjuntos de train, dev y test.

Uso:
    python util/split_data.py --dataset atis --test_size 0.2 --dev_size 0.1 --seed 42
    python util/split_data.py --dataset pibimacec --test_size 0.15 --dev_size 0.1
    
    python util/split_data.py --dataset pibimacecv5 --test_size 0.2 --dev_size 0.15
"""

import argparse
import os
import random
from pathlib import Path


def split_dataset(dataset_name, test_size=0.2, dev_size=0.1, seed=42, skip_label_files=False):
    """
    Divide los datos de la carpeta raw en train, dev y test.
    
    Args:
        dataset_name: Nombre del dataset (ej: 'atis', 'pibimacec')
        test_size: Proporción de datos para test (0.0 a 1.0)
        dev_size: Proporción de datos para dev (0.0 a 1.0)
        seed: Semilla aleatoria para reproducibilidad
    """
    # Configurar rutas
    base_path = Path("data") / dataset_name
    raw_path = base_path / "processed"
    train_path = base_path / "train"
    dev_path = base_path / "dev"
    test_path = base_path / "test"
    
    # Verificar que existe la carpeta raw
    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontró la carpeta: {raw_path}")
    
    # Archivos a procesar
    files = ["label", "seq.in", "seq.out"]
    
    # Verificar que existen todos los archivos
    for filename in files:
        filepath = raw_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    
    # Leer todos los archivos
    print(f"Leyendo archivos de {raw_path}...")
    data = {}
    for filename in files:
        with open(raw_path / filename, 'r', encoding='utf-8') as f:
            data[filename] = [line.rstrip('\n') for line in f]
    
    # Verificar que todos tienen la misma cantidad de líneas
    num_lines = len(data['label'])
    for filename, lines in data.items():
        if len(lines) != num_lines:
            raise ValueError(
                f"Los archivos no tienen la misma cantidad de líneas. "
                f"{filename} tiene {len(lines)} líneas, pero 'label' tiene {num_lines}"
            )
    
    print(f"Total de ejemplos: {num_lines}")
    
    # Crear índices y mezclar
    random.seed(seed)
    indices = list(range(num_lines))
    random.shuffle(indices)
    
    # Calcular tamaños de los conjuntos
    test_count = int(num_lines * test_size)
    dev_count = int(num_lines * dev_size)
    train_count = num_lines - test_count - dev_count
    
    print(f"Train: {train_count} ejemplos ({100*train_count/num_lines:.1f}%)")
    print(f"Dev: {dev_count} ejemplos ({100*dev_size:.1f}%)")
    print(f"Test: {test_count} ejemplos ({100*test_size:.1f}%)")
    
    # Dividir índices
    test_indices = set(indices[:test_count])
    dev_indices = set(indices[test_count:test_count + dev_count])
    train_indices = set(indices[test_count + dev_count:])
    
    # Crear directorios si no existen
    train_path.mkdir(parents=True, exist_ok=True)
    dev_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    # Escribir archivos de train, dev y test
    for filename in files:
        train_lines = [data[filename][i] for i in range(num_lines) if i in train_indices]
        dev_lines = [data[filename][i] for i in range(num_lines) if i in dev_indices]
        test_lines = [data[filename][i] for i in range(num_lines) if i in test_indices]
        
        # Escribir train
        with open(train_path / filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_lines))
            if train_lines:  # Agregar nueva línea al final si hay contenido
                f.write('\n')
        
        # Escribir dev
        with open(dev_path / filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(dev_lines))
            if dev_lines:  # Agregar nueva línea al final si hay contenido
                f.write('\n')
        
        # Escribir test
        with open(test_path / filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_lines))
            if test_lines:  # Agregar nueva línea al final si hay contenido
                f.write('\n')
        
        print(f"✓ {filename}: {len(train_lines)} train, {len(dev_lines)} dev, {len(test_lines)} test")
    
    # Generar archivos de etiquetas únicas (intent_label.txt y slot_label.txt)
    multi_head_labels = any(
        ("," in lbl) for lbl in data['label']
    )
    if skip_label_files or multi_head_labels:
        print("\nOmitiendo generación de intent_label.txt y slot_label.txt "
              + ("(detectado formato multi-head con comas)" if multi_head_labels else "(flag skip)"))
    else:
        print("\nGenerando archivos de etiquetas...")
        unique_intents = sorted(set(data['label']))
        intent_label_file = base_path / "intent_label.txt"
        with open(intent_label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_intents))
            if unique_intents:
                f.write('\n')
        print(f"✓ intent_label.txt: {len(unique_intents)} intents únicos")
        
        unique_slots = set()
        for line in data['seq.out']:
            slots = line.split()
            unique_slots.update(slots)
        unique_slots = sorted(unique_slots)
        
        slot_label_file = base_path / "slot_label.txt"
        with open(slot_label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_slots))
            if unique_slots:
                f.write('\n')
        print(f"✓ slot_label.txt: {len(unique_slots)} slots únicos")
    
    print(f"\n✓ Archivos generados exitosamente en:")
    print(f"  - {train_path}")
    print(f"  - {dev_path}")
    print(f"  - {test_path}")
    if not (skip_label_files or multi_head_labels):
        print(f"  - {base_path} (intent_label.txt, slot_label.txt)")


def main():
    parser = argparse.ArgumentParser(
        description="Divide datos raw en conjuntos de train y test"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Nombre del dataset (ej: 'atis', 'pibimacec')"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proporción de datos para test (default: 0.2)"
    )
    parser.add_argument(
        "--dev_size",
        type=float,
        default=0.1,
        help="Proporción de datos para dev (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad (default: 42)"
    )
    parser.add_argument(
        "--skip_label_files",
        action="store_true",
        help="No generar intent_label.txt ni slot_label.txt (útil en datasets multi-head)",
    )
    
    args = parser.parse_args()
    
    # Validar proporciones
    if not 0 < args.test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1")
    if not 0 < args.dev_size < 1:
        raise ValueError("dev_size debe estar entre 0 y 1")
    if args.test_size + args.dev_size >= 1:
        raise ValueError("La suma de test_size y dev_size debe ser menor a 1")
    
    split_dataset(args.dataset, args.test_size, args.dev_size, args.seed, args.skip_label_files)


if __name__ == "__main__":
    main()
