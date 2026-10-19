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


# ---- Cuadro 2: cardinalidad según span y cabeza sobre el eje de desagregación ------------------
IMACEC = {"indicador": "imacec", "estacionalidad": "original"}
PIB_T = {"frecuencia": "trimestral", "estacionalidad": "original"}


def test_sin_span_ni_cabeza_fila_agregada(cat):
    # la fila agregada figura en varios cuadros (cortes alternativos): una serie, varios pares
    r = cat.resolve_consulta({}, indicador="pib", **PIB_T)
    assert (r.estado, r.modo) == ("resuelto", "agregada")
    assert r.series == ["F032.PIB.FLU.R.CLP.EP18.Z.Z.0.T"] and len(r.cuadros) > 1


def test_span_sin_cabeza_categorias_mencionadas(cat):
    r = cat.resolve_consulta({"activ_cls": 0}, actividad=["mineria", "comercio"], **IMACEC)
    assert (r.estado, r.modo) == ("resuelto", "categorias")
    assert r.candidatos.values("actividad") == {"mineria": 1, "comercio": 1}


def test_cabeza_sin_span_cuadro_completo(cat):
    # «cuánto crecieron las actividades del Imacec»
    r = cat.resolve_consulta({"activ_cls": 1}, **IMACEC)
    assert (r.estado, r.modo) == ("resuelto", "cuadro")
    assert list(r.cuadros) == ["Q001"] and len(r.series) == sum(s["cuadro_id"] == "Q001" for s in cat)
    assert r.destacadas == {}


def test_span_y_cabeza_cuadro_completo_con_destacadas(cat):
    r = cat.resolve_consulta({"activ_cls": 1}, actividad="mineria", **IMACEC)
    assert list(r.cuadros) == ["Q001"] and len(r.series) == sum(s["cuadro_id"] == "Q001" for s in cat)
    assert r.destacadas == {"actividad": ["mineria"]}


def test_doble_desagregacion_region_actividad(cat):
    # activación simultánea: los cuadros región x actividad, uno por región
    r = cat.resolve_consulta({"activ_cls": 1, "reg_cls": 1}, **PIB_T)
    assert r.estado == "resuelto" and len(r.cuadros) == 16
    assert all(s["region_fija"] for s in r.candidatos) and r.candidatos.values("frecuencia") == {"trimestral": len(r.candidatos)}


def test_cada_eje_conserva_su_cardinalidad(cat):
    # «minería por región»: la región se expande (cabeza), la actividad queda en lo mencionado (span)
    r = cat.resolve_consulta({"reg_cls": 1}, actividad="mineria", **PIB_T)
    assert r.estado == "resuelto" and set(r.candidatos.values("actividad")) == {"mineria"}
    assert len(r.series) == len(r.cuadros) > 1


def test_region_mencionada_con_cabeza_de_actividad(cat):
    # «sectores en Valparaíso»: la mención de región selecciona el cuadro por actividad de esa región
    r = cat.resolve_consulta({"activ_cls": 1}, region="valparaiso", **PIB_T)
    assert r.estado == "resuelto" and len(r.cuadros) == 1
    assert r.candidatos.collapse()["region_fija"] == "valparaiso"


def test_combinacion_inexistente(cat):
    assert cat.resolve_consulta({"gast_cls": 1}, region="valparaiso").estado == "no_existe"


def test_cuadro_sin_resolver_pide_aclarar(cat):
    r = cat.resolve_consulta({"activ_cls": 1}, indicador="imacec")
    assert r.estado == "aclarar" and r.aclarar[0] == "estacionalidad"
