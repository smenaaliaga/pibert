"""
Normalización fuera del modelo (paper, sección 4.1).

  - CALCULO: léxico que reconoce las menciones de cálculo y la función de muchos a uno que mapea
    los spans a {nivel, var_anual, var_periodo_anterior}. Lo usa la migración de datos para
    pre-anotar los spans y la inferencia para normalizarlos; la ausencia de mención es `nivel`.
  - REFERENCIA: la compilación sólo se enuncia excepcionalmente y se resuelve fuera del modelo,
    con una regla sobre el texto; sin mención rige la vigente.

Sin dependencias fuera de la biblioteca estándar.
"""
from __future__ import annotations

import re
import unicodedata

CALCULO_VALORES = ["nivel", "var_anual", "var_periodo_anterior"]


def norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
    return s.strip(",.;:?¿!¡()\"'")


# ---- CALCULO ---------------------------------------------------------------------------------
# Cada patrón es una secuencia de regex sobre tokens normalizados; se prueban del más largo al más
# corto. La clase dice qué informa la mención: "pa" (período anterior), "aa" (igual período del año
# anterior) o "var" (variación sin especificar, que se lee como anual); None consume la frase sin
# marcarla.
_ANT = r"(anterior|previo|pasado)"
_PER = r"(periodo|trimestre|mes)"
_DEL = r"(del|de|al|el|a|la)"
_RESP = r"(respecto|relacion|comparacion|comparado|frente|versus|vs|contra)"
_PATRONES_CALCULO: list[tuple[str | None, list[str]]] = [
    # igual período del año anterior
    ("aa", [_RESP, r"(a|al|con)?", r"(el|al)?", r"(mismo|igual)", r"periodo", _DEL, r"ano", _ANT]),
    ("aa", [r"(mismo|igual)", r"periodo", _DEL, r"ano", _ANT]),
    ("aa", [_RESP, r"(a|al|con)?", r"(el|al)?", r"ano", _ANT]),
    ("aa", [r"(en|a)", r"(12|doce)", r"meses"]),
    ("aa", [r"(t/t-12|t-12)"]),
    ("aa", [r"t", r"(vs|versus|contra)", r"t-12"]),
    ("aa", [r"ano", _ANT]),
    ("aa", [r"(interanual\w*|anualizad\w*)"]),
    ("aa", [r"(a|en)", r"(nivel|terminos)", r"anual\w*"]),
    ("aa", [r"anual\w*"]),
    # período inmediatamente anterior
    ("pa", [_RESP, r"(a|al|con)?", r"(el|al)?", _PER, r"(inmediatamente)?", _ANT]),
    ("pa", [_PER, r"(inmediatamente)?", _ANT]),
    ("pa", [r"(en|del)", r"el", r"margen"]),
    ("pa", [r"(t/t-1|t-1)"]),
    ("pa", [r"t", r"(vs|versus|contra|y)", r"t-1"]),
    ("pa", [r"(en|de|del)", r"(el)?", r"(ultimo|corto)", r"(periodo|plazo)"]),
    ("pa", [r"(mensual|trimestral)", r"o", r"(mensual|trimestral)"]),
    ("pa", [r"(des)?aceler\w*"]),
    ("pa", [r"dinami\w*"]),
    ("pa", [r"pulso"]),
    # "contribución al crecimiento", "aporte al desempeño": la mención es de PRESENTACION, no de cálculo
    (None, [r"(al|sobre|en)", r"(el)?", r"(crecimiento|desempeno|resultado)"]),
    # variación sin calificar
    ("var", [r"tasas?", r"de", r"(variacion|crecimiento|cambio)"]),
    ("var", [r"tasas?"]),
    ("var", [r"(variacion|variaciones|vario|variaron|variado|varia)"]),
    ("var", [r"(crecio|crecieron|crecimiento|crece|crecen|crecido)"]),
    ("var", [r"(cambio|cambios|cambiaron|cambiado)"]),
    ("var", [r"(aumento|aumentaron|incremento|incrementos|incrementaron)"]),
    ("var", [r"(movimiento|movio|movieron|movido)"]),
    ("var", [r"(evoluciono|evolucionaron|evolucion)"]),
    ("var", [r"(expandio|expandieron|expansion|contrajo|contracción|contraccion|cayo|cayeron|caida)"]),
    ("var", [r"(avance|avanzo|avanzaron|diferencia|comporto|comportaron|comportamiento|desempeno)"]),
]
_PATRONES_CALCULO = [(c, [re.compile(f"^{p}$") for p in ps])
                     for c, ps in sorted(_PATRONES_CALCULO, key=lambda x: -len(x[1]))]


def _match(pats: list[re.Pattern], toks: list[str], i: int) -> int:
    """Longitud del match de `pats` desde toks[i]; los patrones opcionales ((...)?) pueden saltarse."""
    j = i
    for p in pats:
        if j < len(toks) and p.match(toks[j]):
            j += 1
        elif not p.match(""):
            return 0
    return j - i


def spans_calculo(words: list[str], libres: list[bool] | None = None) -> list[tuple[int, int, str]]:
    """Spans (inicio, fin, clase) de menciones de cálculo, sólo sobre tokens libres (por defecto todos)."""
    toks = [norm(w) for w in words]
    libres = libres or [True] * len(words)
    out, i = [], 0
    while i < len(toks):
        for clase, pats in _PATRONES_CALCULO:
            n = _match(pats, toks, i)
            if n and all(libres[i:i + n]):
                if clase is not None:   # clase None: frase que se consume sin ser mención
                    out.append((i, i + n, clase))
                i += n
                break
        else:
            i += 1
    return out


def clase_calculo(texto: str) -> str | None:
    """Clase ('pa' | 'aa' | 'var') de un span de cálculo, o None si no es una mención conocida."""
    s = spans_calculo(texto.split())
    return s[0][2] if s else None


def normalizar_calculo(spans: list[str]) -> str:
    """Muchos a uno: período anterior manda sobre anual; una variación sin calificar es anual."""
    clases = {clase_calculo(s) for s in spans} - {None}
    if "pa" in clases:
        return "var_periodo_anterior"
    if clases:
        return "var_anual"
    return "nivel"


# ---- REFERENCIA ------------------------------------------------------------------------------
_REFERENCIA_RE = re.compile(r"\b(referencia|base|compilacion(?: de referencia)?)\s+(19\d\d|20\d\d)\b")


def detectar_referencia(texto: str) -> int | None:
    """Compilación de referencia nombrada en la consulta ("base 2013"), o None (rige la vigente)."""
    m = _REFERENCIA_RE.search(" ".join(norm(w) for w in texto.split()))
    return int(m.group(2)) if m else None
