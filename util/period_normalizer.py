import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Dict
from difflib import get_close_matches

import pandas as pd
import dateparser


# ---------------------------
# Utilidades de normalización
# ---------------------------

MONTHS = {
    "enero": 1, "ene": 1,
    "febrero": 2, "feb": 2,
    "marzo": 3, "mar": 3,
    "abril": 4, "abr": 4,
    "mayo": 5, "may": 5,
    "junio": 6, "jun": 6,
    "julio": 7, "jul": 7,
    "agosto": 8, "ago": 8,
    "septiembre": 9, "sep": 9,
    "octubre": 10, "oct": 10,
    "noviembre": 11, "nov": 11,
    "diciembre": 12, "dic": 12,
}

def fuzzy_match_month(word: str, cutoff: float = 0.75) -> Optional[int]:
    """
    Intenta corregir errores ortográficos en nombres de meses.
    
    Args:
        word: Palabra a buscar (ej: "agousto", "feberero")
        cutoff: Umbral de similitud (0-1), por defecto 0.75
        
    Returns:
        Número del mes (1-12) o None si no hay match
    """
    word = word.lower().strip()
    word = strip_accents(word)
    
    # Primero búsqueda exacta
    if word in MONTHS:
        return MONTHS[word]
    
    # Fuzzy matching sobre nombres completos de meses (no abreviaciones)
    full_month_names = [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "setiembre", "octubre", "noviembre", "diciembre"
    ]
    
    matches = get_close_matches(word, full_month_names, n=1, cutoff=cutoff)
    if matches:
        return MONTHS[matches[0]]
    
    return None

RANGE_QUARTER_MONTHS = {
    1: (1, 3),
    2: (4, 6),
    3: (7, 9),
    4: (10, 12),
}

def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = strip_accents(s)
    # Unificar ordinales comunes
    s = re.sub(r"\b1er\b|\b1ero\b|\bprimer\b", "primer", s)
    s = re.sub(r"\b2do\b|\bsegundo\b", "segundo", s)
    s = re.sub(r"\b3er\b|\btercer\b", "tercer", s)
    s = re.sub(r"\b4to\b|\bcuarto\b", "cuarto", s)
    # Normalizar typos en "último" - ultima, ultimo, ultmo, ultim, ulto, ulta -> ultimo
    # ultim?[oa]? = ult + (i)? + (m)? + (o|a)?
    s = re.sub(r"\bult[io]?m?[oa]?\b", "ultimo", s)
    s = re.sub(r"\bultim\b", "ultimo", s)  # "ultim" sin vocal final
    s = re.sub(r"\bultmo\b", "ultimo", s)  # "ultmo" orden incorrecto
    # Reescrituras útiles - tolerando typos en "año"
    s = re.sub(r"\bpresente\s+a[nñ][oiea]\b", "este ano", s)
    s = re.sub(r"\ba[nñ][oiea]\s+en\s+curso\b", "este ano", s)
    s = re.sub(r"\ba[nñ][oiea]\s+actual\b", "este ano", s)
    # Normalizar "mes de cierre" a "ultimo mes del ano"
    s = re.sub(r"\bmes\s+de\s+cierre\s+del\s+ano\b", "ultimo mes del ano", s)
    return s


# ---------------------------
# Modelo de salida
# ---------------------------

@dataclass
class StandardPeriod:
    granularity: str            # "month" | "quarter" | "year"
    start_date: date
    end_date: date              # inclusive
    period_key: str             # "YYYY-MM" | "YYYY-Qn" | "YYYY"
    confidence: str = "high"    # "high" | "medium" | "low"
    source: str = "rules"       # "rules" | "dateparser"


def month_range(year: int, month: int):
    p = pd.Period(f"{year:04d}-{month:02d}", freq="M")
    start = p.start_time.date()
    end = p.end_time.date()
    return start, end

def quarter_range(year: int, q: int):
    p = pd.Period(f"{year:04d}Q{q}", freq="Q")
    start = p.start_time.date()
    end = p.end_time.date()
    return start, end

def year_range(year: int):
    p = pd.Period(f"{year:04d}", freq="Y")
    start = p.start_time.date()
    end = p.end_time.date()
    return start, end


# ---------------------------
# Parsers por reglas
# ---------------------------

def parse_explicit_month(text: str) -> Optional[StandardPeriod]:
    # 1) Formatos numéricos: 08/2024, 08-2024, 2024-08
    m = re.search(r"\b(0?[1-9]|1[0-2])\s*[\/\-]\s*(20\d{2})\b", text)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))
        start, end = month_range(year, month)
        return StandardPeriod("month", start, end, f"{year:04d}-{month:02d}")

    m = re.search(r"\b(20\d{2})\s*[\/\-]\s*(0?[1-9]|1[0-2])\b", text)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        start, end = month_range(year, month)
        return StandardPeriod("month", start, end, f"{year:04d}-{month:02d}")

    # 2) Mes en texto: "agosto 2024", "ago 2024", "mes de abril del 2021"
    m = re.search(
        r"\b(" + "|".join(sorted(MONTHS.keys(), key=len, reverse=True)) + r")\b"
        r"(?:\s+de|\s+del|\s+)?\s*(20\d{2})\b",
        text
    )
    if m:
        mon_str = m.group(1)
        year = int(m.group(2))
        month = MONTHS[mon_str]
        start, end = month_range(year, month)
        return StandardPeriod("month", start, end, f"{year:04d}-{month:02d}")

    # 3) Fuzzy matching para meses con errores ortográficos: "agousto 2024", "del feberero del ano 2023"
    # Patrón 1: "del PALABRA del ano YYYY"
    m = re.search(r"\bdel\s+(\w{4,12})\s+del\s+ano\s+(20\d{2})\b", text)
    if m:
        potential_month = m.group(1)
        year = int(m.group(2))
        month_num = fuzzy_match_month(potential_month, cutoff=0.7)
        if month_num:
            start, end = month_range(year, month_num)
            return StandardPeriod("month", start, end, f"{year:04d}-{month_num:02d}", 
                                confidence="medium", source="fuzzy")
    
    # Patrón 2: "PALABRA YYYY" o "de PALABRA del YYYY"
    m = re.search(r"(?:de|del)?\s*(\w{4,10})\s+(?:de|del)?\s*(20\d{2})\b", text)
    if m:
        potential_month = m.group(1)
        # Evitar falsos positivos con palabras comunes
        if potential_month not in ['ano', 'imacec', 'valor', 'dato', 'datos', 'mes', 'meses']:
            year = int(m.group(2))
            month_num = fuzzy_match_month(potential_month, cutoff=0.7)
            if month_num:
                start, end = month_range(year, month_num)
                return StandardPeriod("month", start, end, f"{year:04d}-{month_num:02d}", 
                                    confidence="medium", source="fuzzy")

    return None


def parse_explicit_quarter(text: str, base_date: Optional[date] = None) -> Optional[StandardPeriod]:
    base = base_date or date.today()
    
    # 1) Notación corta: 1T 2025, 2t2025
    m = re.search(r"\b([1-4])\s*t\s*(20\d{2})\b", text)
    if m:
        q = int(m.group(1))
        year = int(m.group(2))
        start, end = quarter_range(year, q)
        return StandardPeriod("quarter", start, end, f"{year:04d}-Q{q}")
    
    # 1b) Notación corta sin año: 1T, 2t (asume año actual)
    m = re.search(r"\b([1-4])\s*t\b", text)
    if m:
        q = int(m.group(1))
        year = base.year
        start, end = quarter_range(year, q)
        return StandardPeriod("quarter", start, end, f"{year:04d}-Q{q}", confidence="medium")

    # 2) T1 2025, T 1 2025
    m = re.search(r"\bt\s*([1-4])\s*(20\d{2})\b", text)
    if m:
        q = int(m.group(1))
        year = int(m.group(2))
        start, end = quarter_range(year, q)
        return StandardPeriod("quarter", start, end, f"{year:04d}-Q{q}")

    # 3) "trimestre 3 del 2023", "trimestre 3 2023", "3 trimestre 2023"
    # Orden: número + trimestre + año o trimestre + número + año
    m = re.search(r"\b([1-4])\s*trimestre(?:\s+(?:de|del))?\s*(20\d{2})\b", text)
    if m:
        q = int(m.group(1))
        year = int(m.group(2))
        start, end = quarter_range(year, q)
        return StandardPeriod("quarter", start, end, f"{year:04d}-Q{q}")
    
    m = re.search(r"\btrimestre\s*([1-4])(?:\s+(?:de|del))?\s*(20\d{2})\b", text)
    if m:
        q = int(m.group(1))
        year = int(m.group(2))
        start, end = quarter_range(year, q)
        return StandardPeriod("quarter", start, end, f"{year:04d}-Q{q}")

    # 4) "primer/segundo/tercer/cuarto trimestre de/del 2025"
    m = re.search(
        r"\b(primer|segundo|tercer|cuarto)\s+trimestre(?:\s+(?:de|del))?\s*(20\d{2})\b",
        text
    )
    if m:
        ord_map = {"primer": 1, "segundo": 2, "tercer": 3, "cuarto": 4}
        q = ord_map[m.group(1)]
        year = int(m.group(2))
        start, end = quarter_range(year, q)
        return StandardPeriod("quarter", start, end, f"{year:04d}-Q{q}")

    # 4) Rangos por meses: "enero-marzo 2025", "abril a junio 2025"
    m = re.search(
        r"\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
        r"\s*(?:-|a)\s*"
        r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
        r"\s*(20\d{2})\b",
        text
    )
    if m:
        m1 = MONTHS[m.group(1)]
        m2 = MONTHS[m.group(2)]
        year = int(m.group(3))
        # Si coincide exactamente con un trimestre canónico, lo expresamos como quarter
        for q, (qs, qe) in RANGE_QUARTER_MONTHS.items():
            if (m1, m2) == (qs, qe):
                start, end = quarter_range(year, q)
                return StandardPeriod("quarter", start, end, f"{year:04d}-Q{q}")
        # Si no coincide, podrías optar por devolver un rango mensual genérico
        # pero aquí dejamos que otro nivel lo resuelva.
    return None


def parse_explicit_year(text: str) -> Optional[StandardPeriod]:
    m = re.search(r"\b(20\d{2})\b", text)
    # Ojo: esto es muy laxo, por eso solo úsalo si NO detectaste mes/trimestre antes.
    if m:
        year = int(m.group(1))
        start, end = year_range(year)
        return StandardPeriod("year", start, end, f"{year:04d}")
    return None


def parse_relative_periods(text: str, base: date) -> Optional[StandardPeriod]:
    # IMPORTANTE: Reglas MÁS ESPECÍFICAS primero
    
    # "primer trimestre del presente año" (antes de capturar solo "este ano")
    m = re.search(r"\b(primer|segundo|tercer|cuarto)\s+trimestre\s+del\s+(este ano|presente ano|ano actual|ano en curso)\b", text)
    if m:
        ord_map = {"primer": 1, "segundo": 2, "tercer": 3, "cuarto": 4}
        q = ord_map[m.group(1)]
        year = base.year
        start, end = quarter_range(year, q)
        return StandardPeriod("quarter", start, end, f"{year:04d}-Q{q}", source="rules")
    
    # "primer mes del año" (antes de capturar solo "este ano")
    m = re.search(r"\b(primer|segundo|tercer|cuarto)\s+mes\s+del\s+ano\b", text)
    if m:
        ord_to_month = {"primer": 1, "segundo": 2, "tercer": 3, "cuarto": 4}
        month = ord_to_month[m.group(1)]
        year = base.year
        start, end = month_range(year, month)
        return StandardPeriod("month", start, end, f"{year:04d}-{month:02d}", source="rules")

    # Trimestres relativos simples
    if re.search(r"\b(este trimestre|trimestre actual)\b", text):
        p = pd.Period(base.strftime("%Y-%m"), freq="M").asfreq("Q")
        start, end = p.start_time.date(), p.end_time.date()
        q = p.quarter
        return StandardPeriod("quarter", start, end, f"{p.year:04d}-Q{q}", source="rules")

    if re.search(r"\b(trimestre pasado|trimestre anterior)\b", text):
        p = pd.Period(base.strftime("%Y-%m"), freq="M").asfreq("Q") - 1
        start, end = p.start_time.date(), p.end_time.date()
        q = p.quarter
        return StandardPeriod("quarter", start, end, f"{p.year:04d}-Q{q}", source="rules")
    
    # "último trimestre" - asume el trimestre más reciente reportado (trimestre anterior)
    # Después de normalize_text(): "ultimo trimestre" (ya normalizado)
    # trim[ei]?s?tre tolera: trimestre, trimstre, trimetre
    if re.search(r"\bultimo\s+trim[ei]?s?tre\b", text):
        p = pd.Period(base.strftime("%Y-%m"), freq="M").asfreq("Q") - 1
        start, end = p.start_time.date(), p.end_time.date()
        q = p.quarter
        return StandardPeriod("quarter", start, end, f"{p.year:04d}-Q{q}", confidence="medium", source="rules")

    # Mes relativos
    if re.search(r"\b(este mes|mes actual|mes en curso)\b", text):
        start, end = month_range(base.year, base.month)
        return StandardPeriod("month", start, end, f"{base.year:04d}-{base.month:02d}", source="rules")
    if re.search(r"\b(mes pasado|mes anterior|mes previo)\b", text):
        p = pd.Period(base.strftime("%Y-%m"), freq="M") - 1
        start, end = p.start_time.date(), p.end_time.date()
        return StandardPeriod("month", start, end, f"{p.year:04d}-{p.month:02d}", source="rules")
    if re.search(r"\b(mes antepasado)\b", text):
        p = pd.Period(base.strftime("%Y-%m"), freq="M") - 2
        start, end = p.start_time.date(), p.end_time.date()
        return StandardPeriod("month", start, end, f"{p.year:04d}-{p.month:02d}", source="rules")
    
    # "último mes del año" (más específico, antes de capturar solo "ultimo")
    # Nota: después de normalize_text(), "último/última/ultmo/ultim" -> "ultimo"
    if re.search(r"\bultimo\s+mes\s+del\s+ano\b", text):
        year = base.year
        start, end = month_range(year, 12)
        return StandardPeriod("month", start, end, f"{year:04d}-12", confidence="medium", source="rules")
    
    # "último mes" - ya normalizado a "ultimo mes"
    if re.search(r"\bultimo\s+mes\b", text):
        p = pd.Period(base.strftime("%Y-%m"), freq="M") - 1
        start, end = p.start_time.date(), p.end_time.date()
        return StandardPeriod("month", start, end, f"{p.year:04d}-{p.month:02d}", confidence="medium", source="rules")
    
    # Patrones genéricos con "último/última" sin especificar periodo
    # Ejemplos: "el último", "la última", "último dato", "última cifra", "último valor", "última medición"
    # Por defecto asumimos que se refiere al último mes reportado
    if re.search(r"\b(el|la|del|de la)?\s*ultimo\s+(dato|cifra|valor|medicion|reporte|comunicado|registro|informe)?\b", text):
        # Si también menciona "trimestre" o "ano", no usar esta regla (ya se captura arriba)
        if not re.search(r"\b(trimestre|trim[ei]?s?tre|ano)\b", text):
            p = pd.Period(base.strftime("%Y-%m"), freq="M") - 1
            start, end = p.start_time.date(), p.end_time.date()
            return StandardPeriod("month", start, end, f"{p.year:04d}-{p.month:02d}", confidence="medium", source="rules")
    
    # Año relativos (antes de capturar "ultimo" genérico)
    if re.search(r"\b(este ano|ano actual|ano en curso|presente ano)\b", text):
        start, end = year_range(base.year)
        return StandardPeriod("year", start, end, f"{base.year:04d}", source="rules")
    if re.search(r"\b(ano pasado|ano anterior)\b", text):
        start, end = year_range(base.year - 1)
        return StandardPeriod("year", start, end, f"{base.year - 1:04d}", source="rules")
    
    # "último año" - año anterior
    if re.search(r"\bultimo\s+ano\b", text):
        start, end = year_range(base.year - 1)
        return StandardPeriod("year", start, end, f"{base.year - 1:04d}", source="rules")
    
    # "último" o "última" solo, sin contexto adicional
    # Solo aplica si no hay mención de trimestre/año
    if re.search(r"\bultimo\b", text):
        if not re.search(r"\b(trimestre|trim[ei]?s?tre|ano|t[1-4]|[1-4]t)\b", text):
            p = pd.Period(base.strftime("%Y-%m"), freq="M") - 1
            start, end = p.start_time.date(), p.end_time.date()
            return StandardPeriod("month", start, end, f"{p.year:04d}-{p.month:02d}", confidence="low", source="rules")

    # "último reporte del año" / "comunicado del año pasado"
    # Esto no es una fecha directa, sino una referencia documental.
    # Puedes mapearlo a rango anual con baja confianza.
    if re.search(r"\b(ultimo reporte del ano|comunicado del ano pasado)\b", text):
        year = base.year - 1 if "pasado" in text else base.year
        start, end = year_range(year)
        return StandardPeriod("year", start, end, f"{year:04d}", confidence="low", source="rules")

    return None


# ---------------------------
# Fallback con dateparser
# ---------------------------

def parse_with_dateparser(text: str, base: date) -> Optional[StandardPeriod]:
    settings = {
        "RELATIVE_BASE": datetime.combine(base, datetime.min.time()),
        "PREFER_DAY_OF_MONTH": "first",
        "PREFER_DATES_FROM": "past",
    }

    dt = dateparser.parse(text, languages=["es"], settings=settings)
    if not dt:
        return None

    # Heurística: si el texto menciona explícitamente un año y no mes,
    # lo tratamos como año; si menciona un mes en palabras, como mes.
    t = text
    has_month_word = any(re.search(fr"\b{k}\b", t) for k in MONTHS.keys())
    has_year = re.search(r"\b(20\d{2})\b", t) is not None

    if has_month_word and has_year:
        start, end = month_range(dt.year, dt.month)
        return StandardPeriod("month", start, end, f"{dt.year:04d}-{dt.month:02d}", confidence="medium", source="dateparser")

    if has_year and not has_month_word:
        start, end = year_range(dt.year)
        return StandardPeriod("year", start, end, f"{dt.year:04d}", confidence="low", source="dateparser")

    # Si no está claro, asumimos mes del dt.
    start, end = month_range(dt.year, dt.month)
    return StandardPeriod("month", start, end, f"{dt.year:04d}-{dt.month:02d}", confidence="low", source="dateparser")


# ---------------------------
# Función principal
# ---------------------------

def standardize_imacec_time_ref(text: str, base_date: Optional[date] = None) -> Optional[Dict]:
    """
    Devuelve un dict estandarizado con:
      granularity, start_date, end_date, period_key, confidence, source
    """
    base = base_date or date.today()
    t = normalize_text(text)

    # 1) Relativos primero (más específicos: "primer trimestre del presente año")
    p = parse_relative_periods(t, base)
    if p:
        return p.__dict__

    # 2) Reglas explícitas
    p = parse_explicit_quarter(t, base)
    if p:
        return p.__dict__

    p = parse_explicit_month(t)
    if p:
        return p.__dict__

    # 3) Año explícito (solo si no se detectó algo más fino)
    p = parse_explicit_year(t)
    if p:
        return p.__dict__

    # 4) NLP fallback
    p = parse_with_dateparser(t, base)
    if p:
        return p.__dict__

    return None


# ---------------------------
# Ejemplos rápidos
# ---------------------------

if __name__ == "__main__":
    base = date(2025, 12, 9)  # para resultados reproducibles

    samples = [
        # "IMACEC de agosto 2024",
        # "IMACEC 08/2024",
        # "IMACEC 2024-08",
        # "IMACEC del mes pasado",
        # "IMACEC del mes antepasado",
        # "IMACEC primer trimestre del presente año",
        # "IMACEC 2T 2025",
        # "IMACEC abril-junio 2025",
        # "IMACEC del año 2021",
        # "IMACEC del ultimo reporte del año",
        # "valor del comercio del imacec para el primer mes del año",
        # "imacec del agousto del ano 2021",
        # "cuarto trimestre del 2023",
        "imacec para el 3 trimestre 2023",
        "imacec para trimestre 3 del 2023",
        "imacec para trimestre 3 2023",
    ]

    for s in samples:
        print(s, "=>", standardize_imacec_time_ref(s, base))
