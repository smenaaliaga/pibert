import pytest

from data_loader import read_examples
from schema import Schema


def _write(tmp_path, seq_in, label, seq_out):
    d = tmp_path / "t" / "train"
    d.mkdir(parents=True)
    (d / "seq.in").write_text("\n".join(seq_in) + "\n", encoding="utf-8")
    (d / "label").write_text("\n".join(label) + "\n", encoding="utf-8")
    (d / "seq.out").write_text("\n".join(seq_out) + "\n", encoding="utf-8")
    return tmp_path


def test_reads_examples(tmp_path):
    s = Schema.from_catalog()
    root = _write(tmp_path, ["cuanto crecio el imacec"], ["1,0,0"], ["O B-CALCULO O B-INDICADOR"])
    ex = read_examples(str(root), "t", "train", s)
    assert len(ex) == 1
    assert ex[0].head_labels == [1, 0, 0]
    assert ex[0].slot_labels == [0, s.slot_index("B-CALCULO"), 0, s.slot_index("B-INDICADOR")]


def test_unknown_head_label_is_an_error(tmp_path):
    s = Schema.from_catalog()
    root = _write(tmp_path, ["hola"], ["0,2,0"], ["O"])
    with pytest.raises(ValueError, match="'2'"):
        read_examples(str(root), "t", "train", s)


def test_token_tag_mismatch_is_an_error(tmp_path):
    s = Schema.from_catalog()
    root = _write(tmp_path, ["dos palabras"], ["0,0,0"], ["O"])
    with pytest.raises(ValueError, match="etiquetas para"):
        read_examples(str(root), "t", "train", s)


def test_wrong_number_of_heads_is_an_error(tmp_path):
    s = Schema.from_catalog()
    root = _write(tmp_path, ["hola"], ["0,0,0,nivel"], ["O"])
    with pytest.raises(ValueError, match="cabezas"):
        read_examples(str(root), "t", "train", s)
