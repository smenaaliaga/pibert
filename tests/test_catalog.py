"""Casos del paper (sección 4.3-4.4) verificados contra el catálogo construido."""
import pytest

from catalog import DIMS, DIMS_LINGUISTICAS, Catalog


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog.load()


@pytest.fixture(scope="module")
def vigente(cat) -> Catalog:
    return cat.espacio_trabajo()


def test_universo(cat):
    assert len(cat) == 4490
    assert len(cat.cuadros()) == 276
    assert cat.values("indicador") == {"pib": 4393, "imacec": 97}
    assert len(DIMS) == 14 and len(DIMS_LINGUISTICAS) == 8


def test_espacio_trabajo_reduce_a_vigentes(cat, vigente):
    assert len(vigente.cuadros()) == 91
    assert vigente.collapse()["referencia"] == 2018
    assert "tendencia_ciclo" not in vigente.values("estacionalidad")


def test_frecuencia_trimestral_excluye_imacec(vigente):
    assert vigente.filter(frecuencia="trimestral").collapse()["indicador"] == "pib"


def test_frecuencia_mensual_fuerza_imacec(vigente):
    assert vigente.filter(frecuencia="mensual").collapse()["indicador"] == "imacec"


def test_variacion_periodo_anterior_solo_desestacionalizada(vigente):
    # "cuánto aceleró la economía": la contribución m/m sólo se publica sobre series ajustadas
    assert vigente.filter(presentacion="contribucion_pa").collapse()["estacionalidad"] == "desestacionalizada"


def test_resolve_lectura_convencional(cat):
    # "cuánto creció la economía": sólo indicador (por default de dominio) y fila agregada
    r = cat.resolve(indicador="imacec", estacionalidad="original", actividad="total")
    assert r.estado == "resuelto"
    assert r.series == ["F032.IMC.IND.Z.Z.EP18.Z.Z.0.M"]
    assert r.supuestos["presentacion"] == "nivel"     # preferencia aplicada y explicitable
    assert r.colapsadas["frecuencia"] == "mensual"     # propagación: el Imacec sólo es mensual


def test_resolve_preferencias_resuelven_y_se_explicitan(cat):
    # "comercio en Valparaíso": región fija -> cuadro por actividad; nivel y encadenado por preferencia
    r = cat.resolve(region="valparaiso", actividad="comercio")
    assert r.estado == "resuelto"
    assert r.candidatos.values("apertura") == {"actividad": len(r.candidatos)}
    assert r.supuestos == {"presentacion": "nivel", "valoracion": "encadenado"}
    assert r.colapsadas["indicador"] == "pib"


def test_resolve_pide_aclaracion(cat):
    # "PIB de la minería": original y desestacionalizada están ambas publicadas y ninguna
    # preferencia constante las discrimina -> pregunta aclaratoria (tabla condicionada, pendiente)
    r = cat.resolve(indicador="pib", actividad="mineria")
    assert r.estado == "aclarar"
    assert r.aclarar[0] == "estacionalidad"
    assert set(r.aclarar[1]) == {"original", "desestacionalizada"}


def test_resolve_referencia_abre_historico(cat):
    r = cat.resolve(indicador="imacec", referencia=2013, actividad="total", estacionalidad="original")
    assert r.estado in ("resuelto", "aclarar")
    assert r.candidatos.collapse()["vigencia"] == "historico"


def test_resolve_combinacion_inexistente(cat):
    # el Imacec no se publica a precios corrientes
    assert cat.resolve(indicador="imacec", valoracion="corrientes").estado == "no_existe"


def test_lista_equivale_a_cualquiera(cat):
    assert len(cat.filter(frecuencia=["mensual", "trimestral"])) == \
        len(cat.filter(frecuencia="mensual")) + len(cat.filter(frecuencia="trimestral"))


def test_dimension_desconocida(cat):
    with pytest.raises(KeyError):
        cat.filter(periodo="2024")
