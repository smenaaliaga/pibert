from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, List, Pattern, Tuple
from difflib import SequenceMatcher


# =========================
# Configuración base
# =========================

MONTHS = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "setiembre", "octubre", "noviembre", "diciembre"
]

# Alias “semánticos” comunes en lenguaje natural
IMACEC_ALIASES = [
    "imacec",
    "indice mensual de actividad economica",
    "indice de actividad economica mensual",
    "indicador mensual de actividad economica",
    "actividad economica mensual",
    "indice de actividad economica",  # más ambiguo, se desambigua por contexto
]

PIB_ALIASES = [
    "pib",
    "producto interno bruto",
    "producto interior bruto",
    "cuentas nacionales",
]


# =========================
# Utilidades de normalización
# =========================

def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def normalize_text(s: str) -> str:
    """
    Normaliza texto para matching robusto:
    - casefold
    - sin tildes
    - reemplaza todo lo no alfanumérico por espacio
    - compacta espacios
    """
    s = s.strip().casefold()
    s = strip_accents(s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fuzzy_match_word(word: str, candidates: List[str], threshold: float = 0.75) -> Optional[str]:
    """
    Encuentra la mejor coincidencia fuzzy para una palabra en una lista de candidatos.
    Usa SequenceMatcher de difflib para tolerancia a typos.
    
    Args:
        word: Palabra a buscar
        candidates: Lista de palabras candidatas
        threshold: Umbral de similitud (0-1), por defecto 0.75
        
    Returns:
        La palabra candidata con mejor match o None si no supera el umbral
    """
    word_norm = normalize_text(word)
    best_match = None
    best_ratio = 0.0
    
    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        ratio = SequenceMatcher(None, word_norm, candidate_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate
    
    return best_match if best_ratio >= threshold else None


# =========================
# Patrones regex compilados
# =========================

def compile_patterns(patterns: List[str]) -> List[Pattern]:
    return [re.compile(p) for p in patterns]

# Patrones "de superficie" para detectar menciones explícitas
# Incluye variantes con typos comunes
IMACEC_PATTERNS_RAW = [
    r"\bimacec\b",
    r"\bima[cs]e[cs]\b",  # imaces, imasec, imasec
    r"\bi[mn]ac[ae]c\b",  # inacec, imacac, etc
    r"\bindice mensual de actividad economica\b",
    r"\bindice de actividad economica mensual\b",
    r"\bindicador mensual de actividad economica\b",
    r"\bactividad economica mensual\b",
    # patrón más general (ambiguo)
    r"\bindice de actividad economica\b",
]

PIB_PATTERNS_RAW = [
    r"\bpib\b",
    r"\bp[ií]b\b",  # tolera píb
    r"\bproducto interno bruto\b",
    r"\bproducto interior bruto\b",
    r"\bcuentas nacionales\b",
]

IMACEC_PATTERNS = compile_patterns(IMACEC_PATTERNS_RAW)
PIB_PATTERNS = compile_patterns(PIB_PATTERNS_RAW)

# Pistas temporales típicas
IMACEC_TIME_CUES_RAW = [
    r"\bmensual\b",
    r"\bmes\b",
    r"\bultimo mes\b",
    r"\beste mes\b",
    r"\bmes pasado\b",
    r"\bultimo dato\b",
    r"\bmas reciente\b",
] + [rf"\b{m}\b" for m in MONTHS]

PIB_TIME_CUES_RAW = [
    r"\btrimestre\b",
    r"\btrimestral\b",
    r"\banual\b",
    r"\b1t\b|\b2t\b|\b3t\b|\b4t\b",
    r"\bt1\b|\bt2\b|\bt3\b|\bt4\b",
    r"\bprimer trimestre\b|\bsegundo trimestre\b|\btercer trimestre\b|\bcuarto trimestre\b",
    r"\bcuentas nacionales\b",
]

IMACEC_TIME_CUES = compile_patterns(IMACEC_TIME_CUES_RAW)
PIB_TIME_CUES = compile_patterns(PIB_TIME_CUES_RAW)


# =========================
# Configuración de policy
# =========================

@dataclass(frozen=True)
class IndicatorPolicy:
    """
    Ajustes para definir cómo resolver ambigüedades.
    """
    # Si "indice de actividad economica" aparece sin "mensual" ni pista temporal clara,
    # ¿a qué lo mapeamos por defecto?
    ambiguous_activity_index_default: Optional[str] = "imacec"  # o None

    # Si no hay ninguna mención explícita ni pista temporal,
    # no adivinamos el indicador.
    allow_soft_guessing: bool = False


DEFAULT_POLICY = IndicatorPolicy()


# =========================
# Detección principal (sin fuzzy)
# =========================

def _has_any(patterns: List[Pattern], text_norm: str) -> bool:
    return any(p.search(text_norm) for p in patterns)

def detect_indicator(text: str, policy: IndicatorPolicy = DEFAULT_POLICY, use_fuzzy: bool = True) -> Optional[str]:
    """
    Devuelve:
      - "imacec"
      - "pib"
      - None si no está claro

    Orden:
    1) Mención explícita fuerte (PIB primero para evitar colisiones en frases largas).
    2) Mención explícita IMACEC, con tratamiento especial del patrón ambiguo
       'indice de actividad economica'.
    3) Fuzzy matching para detectar typos (si use_fuzzy=True)
    4) Pistas temporales.
    5) Heurísticas suaves opcionales.
    """
    t = normalize_text(text)

    # 1) Mención explícita PIB
    if _has_any(PIB_PATTERNS, t):
        return "pib"

    # 2) Mención explícita IMACEC
    if _has_any(IMACEC_PATTERNS, t):
        # Si la única señal es el patrón ambiguo:
        # 'indice de actividad economica' sin 'mensual'
        if re.search(r"\bindice de actividad economica\b", t) and not re.search(r"\bmensual\b", t):
            # Si hay pistas fuertes PIB, gana PIB
            if _has_any(PIB_TIME_CUES, t):
                return "pib"
            # Si hay pistas IMACEC, gana IMACEC
            if _has_any(IMACEC_TIME_CUES, t):
                return "imacec"
            # Si no hay pistas, aplica policy
            return policy.ambiguous_activity_index_default
        return "imacec"

    # 3) Fuzzy matching para detectar typos en palabras individuales
    if use_fuzzy:
        words = t.split()
        for word in words:
            if len(word) >= 4:  # Solo palabras de 4+ caracteres
                # Buscar similitud con "imacec"
                if fuzzy_match_word(word, ["imacec"], threshold=0.7):
                    return "imacec"
                # Buscar similitud con "pib" (más corta, requiere mayor threshold)
                if len(word) == 3 and fuzzy_match_word(word, ["pib"], threshold=0.66):
                    return "pib"

    # 4) Desambiguación por pistas temporales
    if _has_any(PIB_TIME_CUES, t):
        return "pib"
    if _has_any(IMACEC_TIME_CUES, t):
        return "imacec"

    # 5) Heurísticas suaves opcionales
    if policy.allow_soft_guessing:
        # "actividad economica" + mención de mes
        if "actividad economica" in t and any(m in t for m in MONTHS):
            return "imacec"

    return None


# =========================
# Fuzzy matching (opcional)
# =========================

def _try_import_rapidfuzz():
    try:
        from rapidfuzz import fuzz  # type: ignore
        return fuzz
    except Exception:
        return None

def fuzzy_imacec_score(text: str) -> Optional[int]:
    """
    Score de similitud máximo vs alias IMACEC.
    Requiere rapidfuzz.

    Retorna None si rapidfuzz no está disponible.
    """
    fuzz = _try_import_rapidfuzz()
    if fuzz is None:
        return None

    t = normalize_text(text)
    return max(fuzz.partial_ratio(t, a) for a in IMACEC_ALIASES)

def fuzzy_pib_score(text: str) -> Optional[int]:
    """
    Score de similitud máximo vs alias PIB.
    Requiere rapidfuzz.

    Retorna None si rapidfuzz no está disponible.
    """
    fuzz = _try_import_rapidfuzz()
    if fuzz is None:
        return None

    t = normalize_text(text)
    return max(fuzz.partial_ratio(t, a) for a in PIB_ALIASES)

def detect_indicator_with_fuzzy(
    text: str,
    policy: IndicatorPolicy = DEFAULT_POLICY,
    imacec_threshold: int = 85,
    pib_threshold: int = 85,
) -> Optional[str]:
    """
    Extiende detect_indicator con un fallback fuzzy.
    - Primero intenta la detección determinística.
    - Luego usa scores fuzzy con umbrales altos.
    - Aplica pistas temporales como “freno de seguridad”.

    Requiere rapidfuzz para que el fallback funcione.
    """
    base = detect_indicator(text, policy=policy)
    if base:
        return base

    # Fuzzy disponible?
    s_i = fuzzy_imacec_score(text)
    s_p = fuzzy_pib_score(text)
    if s_i is None or s_p is None:
        return None

    t = normalize_text(text)

    # Si hay pistas temporales muy fuertes, impón eso
    if _has_any(PIB_TIME_CUES, t):
        return "pib"
    if _has_any(IMACEC_TIME_CUES, t):
        return "imacec"

    # Si ambos pasan umbral, decide por mayor score
    i_ok = s_i >= imacec_threshold
    p_ok = s_p >= pib_threshold

    if i_ok and p_ok:
        return "imacec" if s_i >= s_p else "pib"
    if i_ok:
        return "imacec"
    if p_ok:
        return "pib"

    return None


# =========================
# Estandarización de menciones
# =========================

# Reemplazos conservadores sobre el texto original
IMACEC_REWRITE_RULES: List[Tuple[Pattern, str]] = [
    (re.compile(r"\bIMACEC\b", flags=re.IGNORECASE), "{canonical}"),
    (re.compile(r"(?i)\bíndice\s+mensual\s+de\s+actividad\s+económica\b"), "{canonical}"),
    (re.compile(r"(?i)\bíndice\s+de\s+actividad\s+económica\s+mensual\b"), "{canonical}"),
    (re.compile(r"(?i)\bindicador\s+mensual\s+de\s+actividad\s+económica\b"), "{canonical}"),
    (re.compile(r"(?i)\bactividad\s+económica\s+mensual\b"), "{canonical}"),
]

def standardize_imacec_mentions(
    text: str,
    canonical: str = "imacec",
    use_fuzzy: bool = False,
    policy: IndicatorPolicy = DEFAULT_POLICY,
) -> str:
    """
    Reemplaza expresiones típicas de IMACEC por forma canónica.
    - Solo actúa si el indicador detectado es IMACEC.
    - No intenta reemplazar PIB.

    use_fuzzy=True activa detect_indicator_with_fuzzy.
    """
    detector = detect_indicator_with_fuzzy if use_fuzzy else detect_indicator
    ind = detector(text, policy=policy)  # type: ignore

    if ind != "imacec":
        return text

    out = text
    for pat, repl in IMACEC_REWRITE_RULES:
        out = pat.sub(repl.format(canonical=canonical), out)

    return out


# =========================
# API de alto nivel
# =========================

def standardize_indicator(
    text: str,
    canonical_imacec: str = "imacec",
    canonical_pib: str = "pib",
    use_fuzzy: bool = False,
    policy: IndicatorPolicy = DEFAULT_POLICY,
) -> dict:
    """
    Devuelve un dict útil para pipelines:
    {
        "indicator": "imacec" | "pib" | None,
        "text_standardized_imacec": <texto con menciones IMACEC normalizadas>,
        "text_norm": <texto normalizado>
    }
    """
    t_norm = normalize_text(text)
    detector = detect_indicator_with_fuzzy if use_fuzzy else detect_indicator
    ind = detector(text, policy=policy)  # type: ignore

    std_imacec = standardize_imacec_mentions(
        text,
        canonical=canonical_imacec,
        use_fuzzy=use_fuzzy,
        policy=policy
    )

    # (Opcional) Si quisieras estandarizar PIB, podrías agregar reglas similares.
    # Por ahora lo dejamos explícito:
    return {
        "indicator": ind if ind in (canonical_imacec, canonical_pib, "imacec", "pib") else ind,
        "text_standardized_imacec": std_imacec,
        "text_norm": t_norm,
    }


# =========================
# Demo rápida / tests simples
# =========================

def _demo():
    samples = [
        "¿Cuánto fue el IMACEC de agosto 2024?",
        "Dame el índice mensual de actividad económica más reciente",
        "¿Cuál es la variación mensual de la actividad económica?",
        "¿Cuál fue el PIB del 2T 2025?",
        "PIB anual 2024",
        "¿Cómo va la economía?",
        "Índice de actividad económica de marzo 2025",
        "Índice de actividad económica del tercer trimestre 2023",
        "actividad económica mensual de septiembre",
        "producto interno bruto del primer trimestre",
    ]

    print("=== DEMO detect_indicator ===")
    for s in samples:
        print(f"- {s}\n  -> {detect_indicator(s)}")

    print("\n=== DEMO standardize_imacec_mentions ===")
    for s in samples:
        print(f"- {s}\n  -> {standardize_imacec_mentions(s)}")

    print("\n=== DEMO API alto nivel ===")
    for s in samples:
        print(f"- {s}\n  -> {standardize_indicator(s)}")


if __name__ == "__main__":
    _demo()