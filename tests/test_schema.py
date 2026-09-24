import json

import pytest

from schema import CABEZAS_LEGADO, Schema


def test_legacy_schema_from_data():
    s = Schema.from_split_files("data/pibimacecv3", CABEZAS_LEGADO)
    assert s.head_names == CABEZAS_LEGADO
    assert s.heads["calc_mode"] == ["contribution", "original", "prev_period", "yoy"]
    assert s.heads["activity"] == ["general", "none", "specific"]
    assert s.entities == ["ACTIVITY", "FREQUENCY", "INDICATOR", "INVESTMENT", "PERIOD", "REGION", "SEASONALITY"]
    assert s.slot_labels[0] == "O" and len(s.slot_labels) == 1 + 2 * len(s.entities)


def test_catalog_schema():
    s = Schema.from_catalog()
    assert s.head_names == ["activ_cls", "reg_cls", "gast_cls", "calculo"]
    assert s.entities == ["INDICADOR", "FRECUENCIA", "ESTACIONALIDAD", "PRESENTACION", "VALORACION",
                          "ACTIVIDAD", "REGION", "GASTO", "PERIODO", "REFERENCIA"]
    assert s.slot_index("B-GASTO") == 1 + 2 * s.entities.index("GASTO")


def test_unknown_labels_raise():
    s = Schema.from_catalog()
    with pytest.raises(ValueError):
        s.head_index("calculo", "yoy")
    with pytest.raises(ValueError):
        s.slot_index("B-INVESTMENT")


def test_roundtrip(tmp_path):
    s = Schema.from_catalog()
    p = s.save(tmp_path)
    t = Schema.load(p)
    assert t.to_dict() == s.to_dict() and t.fingerprint() == s.fingerprint()
    assert json.loads(p.read_text())["version"] == "catalogo"
