import re
import unicodedata
from datetime import date
from typing import Optional, Dict, Any, Tuple

# ---------------------------
# Normalización / utilidades
# ---------------------------

_MONTHS = {
    "ene": 1, "enero": 1,
    "feb": 2, "febrero": 2, "feberero": 2,  # feberero: typo común
    "febero": 2, "febro": 2, "febr": 2,
    "mar": 3, "marzo": 3, "marz": 3, "marxo": 3,
    "abr": 4, "abril": 4,
    "may": 5, "mayo": 5, "mallo": 5,
    "jun": 6, "junio": 6, "juno": 6,
    "jul": 7, "julio": 7, "jullio": 7,
    "ago": 8, "agosto": 8, "agosoto": 8, "agoto": 8,
    "sep": 9, "sept": 9, "septiembre": 9, "septimbre": 9, "septiebre": 9,
    "set": 9, "setiembre": 9, "setimbre": 9,
    "oct": 10, "octubre": 10, "octube": 10, "octbre": 10, "ocubre": 10,
    "nov": 11, "noviembre": 11, "noviemre": 11, "nobiembre": 11,
    "dic": 12, "diciembre": 12, "dicimebre": 12, "diciembe": 12, "dicimbre": 12,
}

_REL_YEAR = {
    # actual / vigente
    "actual": 0,
    "vigente": 0,
    "curso": 0,         # "en curso"
    "presente": 0,
    "este": 0,          # "este año"
    # pasado
    "pasado": -1,
    "anterior": -1,
}

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def _norm_text(s: str) -> str:
    s = _strip_accents(s.lower())
    # unifica separadores “raros” a espacio o slash
    s = s.replace("\\", "/")
    s = re.sub(r"[,\.;:\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _parse_year(y: str, ref_year: int, pivot: int = 70) -> Optional[int]:
    """
    - y puede ser: '2024', '24', 'actual', 'pasado', 'anterior', etc.
    - pivot: 2-dígitos >= pivot => 1900s; < pivot => 2000s
    """
    if y is None:
        return None
    y = y.strip().lower()

    # relativo: "actual", "vigente", "pasado", "anterior", etc.
    if y in _REL_YEAR:
        return ref_year + _REL_YEAR[y]

    # casos como "en curso" o "este ano"
    if y in ("en curso",):
        return ref_year
    if y in ("este ano", "este año", "ano actual", "año actual", "ano en curso", "año en curso",
             "ano vigente", "año vigente", "ano presente", "año presente"):
        return ref_year

    # numérico
    if re.fullmatch(r"\d{4}", y):
        return int(y)

    if re.fullmatch(r"\d{2}", y):
        yy = int(y)
        return (1900 + yy) if yy >= pivot else (2000 + yy)

    return None

def _parse_month(m: str) -> Optional[int]:
    if m is None:
        return None
    m = m.strip().lower()
    if m.isdigit():
        mi = int(m)
        if 1 <= mi <= 12:
            return mi
        return None
    return _MONTHS.get(m)

def _mk_result(year: int, month: int) -> Dict[str, Any]:
    return {
        "year": year,
        "month": month,
        "period": f"{year:04d}-{month:02d}",
        "date": date(year, month, 1),
    }


def _add_months(d: date, delta: int) -> date:
    m0 = d.month - 1 + delta
    y = d.year + m0 // 12
    m = m0 % 12 + 1
    return date(y, m, 1)


def _mk_range(y1: int, m1: int, y2: int, m2: int, source: str) -> Dict[str, Any]:
    return {
        "start": _mk_result(y1, m1),
        "end": _mk_result(y2, m2),
        "source": source,
    }

# ---------------------------
# Regex principales
# ---------------------------

# Mes por nombre (ene/enero, feb/febrero/feberero, etc.) con variantes comunes
_MONTH_RX = r"(?:"
_MONTH_RX += r"ene(?:ro)?|"
_MONTH_RX += r"feb(?:rero)?|feberero|febero|febro|febr|"
_MONTH_RX += r"mar(?:zo)?|marz|marxo|"
_MONTH_RX += r"abr(?:il)?|"
_MONTH_RX += r"may(?:o)?|mallo|"
_MONTH_RX += r"jun(?:io)?|juno|"
_MONTH_RX += r"jul(?:io)?|jullio|"
_MONTH_RX += r"ago(?:sto)?|agosoto|agoto|"
_MONTH_RX += r"sep(?:t(?:iembre)?)?|septimbre|septiebre|"
_MONTH_RX += r"set(?:iembre)?|setimbre|"
_MONTH_RX += r"oct(?:ubre)?|octube|octbre|ocubre|"
_MONTH_RX += r"nov(?:iembre)?|noviemre|nobiembre|"
_MONTH_RX += r"dic(?:iembre)?|dicimebre|diciembe|dicimbre"
_MONTH_RX += r")"

# Tokens de "año ..."
# Ej: "del ano actual", "del año pasado", "del año en curso", "de este año"
_YEAR_TAIL_RX = r"(?:del?\s+)?(?:ano|año)\s+(?:actual|vigente|presente|pasado|anterior|en\s+curso|este)"

_PATTERNS: Tuple[re.Pattern, ...] = (
    # 1) YYYY-MM o YYYY/MM (con o sin 'periodo', 'mes', etc.)
    re.compile(rf"(?:\b(?:periodo|per[ií]odo|mes)\b\s*)?(?P<y>\d{{4}})\s*[-/]\s*(?P<m>0?[1-9]|1[0-2])\b"),

    # 2) MM-YYYY o MM/YYYY (con o sin 'mes', etc.)
    re.compile(rf"(?:\b(?:periodo|per[ií]odo|mes)\b\s*)?(?P<m>0?[1-9]|1[0-2])\s*[-/]\s*(?P<y>\d{{2,4}})\b"),

    # 3) MesNombre[-/]YYYY o MesNombre YYYY o MesNombre del 24
    re.compile(rf"\b(?P<mon>{_MONTH_RX})\b\s*(?:[-/ ]\s*)?(?:(?:de|del)?\s*)?(?P<y>\d{{2,4}})\b"),

    # 4) YYYY MesNombre  (ej: '2024 ene')
    re.compile(rf"\b(?P<y>\d{{4}})\b\s+(?P<mon>{_MONTH_RX})\b"),

    # 5) MesNombre + 'año actual/pasado/anterior/en curso/este...'
    # ej: "diciembre del año actual", "enero del año pasado", "febrero del año en curso"
    re.compile(rf"\b(?P<mon>{_MONTH_RX})\b\s+(?P<ytail>{_YEAR_TAIL_RX})\b"),

    # 6) Solo MesNombre (sin año) → asumir año de referencia
    re.compile(rf"\b(?P<mon>{_MONTH_RX})\b"),
)

def extract_month_period(text: str, reference_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
    """
    Extrae el primer periodo mensual detectado en 'text'.
    Si no se menciona año, asume año actual (reference_date.year).
    """
    if reference_date is None:
        reference_date = date.today()

    s = _norm_text(text)

    for rx in _PATTERNS:
        m = rx.search(s)
        if not m:
            continue

        gd = m.groupdict()

        # ---- CASOS CON MES POR NOMBRE ----
        if gd.get("mon"):
            month = _parse_month(gd["mon"])
            if month is None:
                continue

            # 1) Mes + año explícito (enero 2024 / marzo del 24)
            if gd.get("y"):
                year = _parse_year(gd["y"], reference_date.year)
                if year is None:
                    continue
                return _mk_result(year, month)

            # 2) Mes + año relativo (año actual / pasado / en curso)
            if gd.get("ytail"):
                ytail = gd["ytail"].replace("en curso", "curso")
                rel = None
                for key in _REL_YEAR.keys():
                    if re.search(rf"\b{key}\b", ytail):
                        rel = key
                        break
                if rel is None and re.search(r"\beste\b", ytail):
                    rel = "este"
                if rel is None:
                    rel = "actual"

                year = _parse_year(rel, reference_date.year)
                return _mk_result(year, month)

            # 3) 🔥 NUEVO: Mes SIN año → asumir año actual
            return _mk_result(reference_date.year, month)

        # ---- CASOS NUMÉRICOS (YYYY-MM / MM-YYYY) ----
        year_raw = gd.get("y")
        month_raw = gd.get("m")
        year = _parse_year(year_raw, reference_date.year)
        month = _parse_month(month_raw)
        if year is None or month is None:
            continue

        return _mk_result(year, month)

    return None


def extract_month_range(text: str, reference_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
    """
    Detecta un rango de meses (inicio/fin). Asume que el texto describe un rango.
    Si falta algún extremo, usa heurísticas con la fecha de referencia.
    """
    if reference_date is None:
        reference_date = date.today()

    ref_month = date(reference_date.year, reference_date.month, 1)
    s = _norm_text(text)

    def _py(y: str) -> Optional[int]:
        return _parse_year(y, reference_date.year)

    # 0) "entre MesNombre y MesNombre [del/de] YYYY"
    # Ej: "entre enero y febrero del 2023"
    m = re.search(rf"entre\s+(?P<mon1>{_MONTH_RX})\b\s*(?:y|e)\s+(?P<mon2>{_MONTH_RX})\b\s*(?:del?\s+)?(?P<y>\d{{2,4}})", s)
    if m:
        m1 = _parse_month(m.group("mon1")); m2 = _parse_month(m.group("mon2"))
        y = _py(m.group("y"))
        if None not in (m1, m2, y):
            return _mk_range(y, m1, y, m2, "entre_mes_y_mes")

    # 1) MM-YYYY a MM-YYYY (o con /)
    m = re.search(r"(?:mes\s+)?(?P<m1>0?[1-9]|1[0-2])[-/](?P<y1>\d{2,4})\s*(?:a|al|hasta|entre|y|\-|–|—)\s*(?:mes\s+)?(?P<m2>0?[1-9]|1[0-2])[-/](?P<y2>\d{2,4})", s)
    if m:
        y1 = _py(m.group("y1")); y2 = _py(m.group("y2"))
        m1 = _parse_month(m.group("m1")); m2 = _parse_month(m.group("m2"))
        if None not in (y1, y2, m1, m2):
            return _mk_range(y1, m1, y2, m2, "mm_yyyy_range")

    # 2) YYYY-MM a YYYY-MM
    m = re.search(r"\b(?P<y1>\d{4})[-/](?P<m1>0?[1-9]|1[0-2])\b\s*(?:a|al|hasta|entre|y|\-|–|—)\s*\b(?P<y2>\d{4})[-/](?P<m2>0?[1-9]|1[0-2])\b", s)
    if m:
        y1 = _py(m.group("y1")); y2 = _py(m.group("y2"))
        m1 = _parse_month(m.group("m1")); m2 = _parse_month(m.group("m2"))
        if None not in (y1, y2, m1, m2):
            return _mk_range(y1, m1, y2, m2, "yyyy_mm_range")

    # 3) MesNombre [año] a MesNombre [año]
    m = re.search(rf"\b(?P<mon1>{_MONTH_RX})\b\s*(?P<y1>\d{{2,4}})?\s*(?:a|al|hasta|entre|y)\s*\b(?P<mon2>{_MONTH_RX})\b\s*(?P<y2>\d{{2,4}})?", s)
    if m:
        m1 = _parse_month(m.group("mon1")); m2 = _parse_month(m.group("mon2"))
        y1 = _py(m.group("y1")) if m.group("y1") else None
        y2 = _py(m.group("y2")) if m.group("y2") else None
        if y1 is None and y2 is not None:
            y1 = y2
        if y2 is None and y1 is not None:
            y2 = y1
        if y1 is None and y2 is None:
            y1 = y2 = reference_date.year
        if None not in (m1, m2, y1, y2):
            return _mk_range(y1, m1, y2, m2, "month_name_range")

    # 4) Rango de años (asume meses completos)
    m = re.search(r"\b(?P<y1>\d{4})\b\s*(?:a|al|hasta|entre|y|\-|–|—)\s*(?P<y2>\d{4})\b", s)
    if m:
        y1 = _py(m.group("y1")); y2 = _py(m.group("y2"))
        if None not in (y1, y2):
            return _mk_range(y1, 1, y2, 12, "year_range")

    # 5) Ultimos/desde hace N meses
    m = re.search(r"ultim(?:o|os)\s+(?P<n>\d+)\s+mes", s) or re.search(r"desde\s+hace\s+(?P<n>\d+)\s+mes", s) or re.search(r"(?P<n>\d+)\s+mes(?:es)?\s+atras", s)
    if m:
        n = int(m.group("n"))
        if n > 0:
            end_d = ref_month
            start_d = _add_months(end_d, -(n - 1))
            return _mk_range(start_d.year, start_d.month, end_d.year, end_d.month, "relative_months")

    # 6) Ultimo ano disponible / ultimo año
    if re.search(r"ultimo\s+ano", s):
        end_d = ref_month
        start_d = _add_months(end_d, -11)
        return _mk_range(start_d.year, start_d.month, end_d.year, end_d.month, "last_12m")

    # 7) A partir de <fecha> (usa fin = referencia)
    m = re.search(r"a\s+partir\s+de\s+(?P<y>\d{4})(?:[-/](?P<m>0?[1-9]|1[0-2]))?", s)
    if m:
        y1 = _py(m.group("y"))
        m1 = _parse_month(m.group("m")) if m.group("m") else 1
        if None not in (y1, m1):
            return _mk_range(y1, m1, ref_month.year, ref_month.month, "from_date_to_ref")

    # 8) Desde <mes> [año] en adelante (fin = referencia)
    m = re.search(rf"desde\s+(?P<mon>{_MONTH_RX})\s*(?P<y>\d{{2,4}})?", s)
    if m:
        m1 = _parse_month(m.group("mon"))
        y1 = _py(m.group("y")) if m.group("y") else reference_date.year
        if None not in (y1, m1):
            return _mk_range(y1, m1, ref_month.year, ref_month.month, "from_month_to_ref")

    # 9) Hasta <fecha> (sin inicio explicito, asume ventana 12m)
    m = re.search(rf"hasta\s+(?:(?P<m>0?[1-9]|1[0-2])[-/](?P<y>\d{{2,4}})|(?P<mon>{_MONTH_RX})\s*(?P<y2>\d{{2,4}})|(?P<y3>\d{{4}}))", s)
    if m:
        if m.group("mon"):
            m2 = _parse_month(m.group("mon"))
            y2 = _py(m.group("y2")) if m.group("y2") else reference_date.year
        elif m.group("m"):
            m2 = _parse_month(m.group("m"))
            y2 = _py(m.group("y"))
        else:
            m2 = 12
            y2 = _py(m.group("y3"))
        if None not in (m2, y2):
            end_d = date(y2, m2, 1)
            start_d = _add_months(end_d, -11)
            return _mk_range(start_d.year, start_d.month, end_d.year, end_d.month, "to_date_last_12m")

    return None


# ---------------------------
# Mini-tests rápidos (opcional)
# ---------------------------
if __name__ == "__main__":
    ref = date(2026, 1, 8)

    samples = [
        "en diciembreoa: diciembre del año actual cuanto vario ...",
        "enero 24 cuanto vario ...",
        "feb 2024 dame la variacion ...",
        "marzo del 24 cuanto fue ...",
        "mes 01-2024 cuanto vario ...",
        "periodo 2024-01 cual fue ...",
        "2024/01 dame la variacion ...",
        "2024-01 cuanto cambio ...",
        "enero del año pasado cuanto vario ...",
        "mayo de 2019 dame ...",
        "2019/05 cuanto vario ...",
        "2024 ene cuanto vario ...",
        "ene-2024 cual fue ...",
        "mes 05/2019 cuanto crecio ...",
        "en febrero ..."
    ]

    for t in samples:
        print(t, "->", extract_month_period(t, reference_date=ref))
