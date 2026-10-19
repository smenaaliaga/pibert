"""Normalización de CALCULO y detección de REFERENCIA (paper, sección 4.1)."""
import pytest

from normalizacion import detectar_referencia, normalizar_calculo, spans_calculo
from schema import Schema


def _spans(texto):
    w = texto.split()
    return [" ".join(w[a:b]) for a, b, _ in spans_calculo(w)]


def test_ejemplo_del_paper():
    # «cuánto creció el producto interno bruto sin estacionalidad»: «creció» -> var_anual
    s = _spans("cuánto creció el producto interno bruto sin estacionalidad")
    assert s == ["creció"] and normalizar_calculo(s) == "var_anual"


@pytest.mark.parametrize("texto, esperado", [
    ("cuánto aceleró la economía", "var_periodo_anterior"),
    ("variación respecto al periodo anterior del pib", "var_periodo_anterior"),
    ("cómo se movieron t vs t-1 los sectores", "var_periodo_anterior"),
    ("variación respecto al mismo periodo del año anterior", "var_anual"),
    ("crecimiento a 12 meses del imacec", "var_anual"),
    ("variación interanual del pib", "var_anual"),
    ("dame el pib del último trimestre", "nivel"),
    ("contribución de la minería al crecimiento del pib", "nivel"),   # PRESENTACION, no cálculo
])
def test_normalizacion_muchos_a_uno(texto, esperado):
    assert normalizar_calculo(_spans(texto)) == esperado


def test_solo_tokens_libres():
    w = "variación interanual".split()
    assert spans_calculo(w, [True, False]) == [(0, 1, "var")]


def test_referencia_fuera_del_modelo():
    assert detectar_referencia("pib anual base 2013") == 2013
    assert detectar_referencia("pib de la compilación de referencia 2008") == 2008
    assert detectar_referencia("pib del 2013") is None
    assert "REFERENCIA" not in Schema.from_catalog().entities


def test_dataset_v5_sigue_el_esquema_del_paper():
    from data_loader import read_examples
    s = Schema.load("data/pibimacecv5")
    assert s.to_dict() == Schema.from_catalog().to_dict()
    for split in ("train", "dev", "test"):
        assert read_examples("data", "pibimacecv5", split, s)
