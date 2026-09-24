"""
Construye el catálogo de resolución de PIBERT a partir de la exportación de la BDE.

La tupla de cada serie se DERIVA de su código: cada segmento del identificador
codifica una dimensión (indicador, presentación, valoración, referencia, actividad,
región, estacionalidad, frecuencia...). El archivo de facetas por cuadro se usa
sólo para las dimensiones que el código no codifica (apertura) y para validar.

Entradas (data/catalogo/raw/):
  - catalogo_series_bde_<fecha>.xlsx, hoja "Acotacion": universo PIB + Imacec
  - cuadros_facetas_bde.xlsx, hoja "Cuadros_facetas": facetas por cuadro

Salidas (data/catalogo/):
  - series.csv         una fila por par cuadro-serie con la tupla completa
  - cuadros.csv        una fila por cuadro con la tupla agregada
  - vocabularios.json  valores canónicos por dimensión (con conteos)
  - reporte.md         validaciones: colisiones, códigos sin mapear, dispersión

Uso:
    python util/build_catalog.py [--series-xlsx RUTA] [--facetas-xlsx RUTA] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --------------------------------------------------------------------------- #
# Estructura de los códigos de serie (verificada contra las facetas de cuadro)
#
#   F032 (10 seg): F032.{ind}.{tipo}.{val}.{mon}.{hist}.{act}.{subact}.{sa}.{freq}
#   F033 ( 8 seg): F033.{gasto}.{tipo}.{val}.{mon}.{hist}.{sa}.{freq}
#   F035 (12 seg): F035.{ind}.{tipo}.{val}.{mon}.{hist}.{act}.{subact}.{sub2}.{region}.{sa}.{freq}
# --------------------------------------------------------------------------- #

FAMILIAS = {
    "F032": dict(n=10, ind=1, tipo=2, val=3, hist=5, act=(6, 7, None), region=None, sa=8, freq=9),
    "F033": dict(n=8, ind=None, tipo=2, val=3, hist=5, act=None, region=None, sa=6, freq=7, gasto=1),
    "F035": dict(n=12, ind=1, tipo=2, val=3, hist=5, act=(6, 7, 8), region=9, sa=10, freq=11),
}

PRESENTACION = {
    "FLU": "nivel",            # flujo en nivel
    "IND": "nivel",            # índice (Imacec)
    "V12": "contribucion_aa",  # contribución respecto de igual período del año anterior
    "VAR": "contribucion_pa",  # contribución respecto del período anterior
    "RPA": "participacion",    # participación porcentual
    "TAS": "tasa",             # tasa (inversión / PIB)
}

ESTACIONALIDAD = {"0": "original", "1": "desestacionalizada", "3": "desestacionalizada", "4": "tendencia_ciclo"}
FRECUENCIA = {"M": "mensual", "T": "trimestral", "A": "anual"}

# hist: año de compilación de referencia / base, más flags de empalme e info histórica
def decode_hist(seg: str) -> tuple[int, bool, bool]:
    if seg == "HIST":
        return 2008, False, True
    if seg == "HIST13":
        return 2013, False, True
    if seg.startswith("EP"):
        return 2000 + int(seg[2:]), True, False
    return int(seg), False, False


# Actividad económica: (act, subact[, sub2]) -> slug. Se busca de lo más específico a lo más general.
ACTIVIDAD = {
    ("Z",): "total",
    ("CONT",): "total",
    ("01",): "agropecuario_silvicola",
    ("02",): "pesca",
    ("03",): "mineria",
    ("03", "05"): "mineria_cobre",
    ("03", "06"): "otras_actividades_mineras",
    ("04",): "industria_manufacturera",
    ("04", "07"): "alimentos_bebidas_tabaco",
    ("04", "08"): "textil_prendas_cuero",
    ("04", "09"): "maderas_muebles",
    ("04", "10"): "celulosa_papel_imprentas",
    ("04", "11"): "quimica_petroleo_caucho_plastico",
    ("04", "12"): "minerales_no_metalicos_metalica_basica",
    ("04", "14"): "productos_metalicos_maquinaria_equipo",
    ("04", "28"): "alimentos",
    ("04", "29"): "bebidas_tabaco",
    ("04", "30"): "refinacion_petroleo",
    ("04", "31"): "quimica_caucho_plastico",
    ("05",): "electricidad_gas_agua",
    ("06",): "construccion",
    ("07",): "comercio_restaurantes_hoteles",
    ("07", "32"): "comercio",
    ("07", "33"): "restaurantes_hoteles",
    ("08",): "transporte_comunicaciones",
    ("08", "19"): "transporte",
    ("08", "20"): "comunicaciones",
    ("09",): "servicios_financieros_empresariales",
    ("09", "21", "27"): "servicios_financieros",
    ("09", "34"): "servicios_financieros",
    ("09", "35"): "servicios_empresariales",
    ("10",): "servicios_vivienda",
    ("11",): "servicios_personales",
    ("12",): "administracion_publica",
    ("13",): "imputaciones_bancarias",
    ("14",): "iva",
    ("1415",): "impuestos_productos",
    ("15",): "derechos_importacion",
    ("16",): "costo_factores",
    ("17",): "subtotal",
    ("N03",): "no_minero",
    ("COM",): "comercio",
    ("PB",): "produccion_bienes",
    ("RB",): "resto_bienes",
    ("RH",): "restaurantes_hoteles",
    ("SERV",): "servicios",
    ("IMP",): "impuestos_productos",
}

# Segundo segmento en F032/F035 que no es el indicador sino una fila especial del cuadro
FILAS_ESPECIALES = {
    "ICF": "costo_factores",           # Imacec a costo de factores
    "IMB": "imputaciones_bancarias",   # regional
    "IMP": "iva_derechos_importacion", # regional
    "PCF": "subtotal_regionalizado",   # regional
}

GASTO = {
    "PIB": "pib",
    "DDI": "demanda_interna", "DIR": "resto_demanda_interna",
    "CTO": "consumo_total", "CPR": "consumo_hogares", "CDU": "bienes_durables",
    "CND": "bienes_no_durables", "CSE": "servicios", "COG": "consumo_gobierno",
    "FBK": "formacion_bruta_capital", "FKF": "fbcf", "FKK": "construccion_otras_obras",
    "FKO": "maquinaria_equipo", "FKI": "maquinaria_equipo_importada", "FKN": "maquinaria_equipo_nacional",
    "VAX": "variacion_existencias",
    "XBS": "exportaciones", "XBI": "exportaciones_bienes", "XSE": "exportaciones_servicios",
    "XAP": "exportaciones_agropecuario_silvicola_pesca", "XAS": "exportaciones_agropecuario_silvicola",
    "XPE": "exportaciones_pesca", "XMI": "exportaciones_mineria", "XCO": "exportaciones_cobre",
    "XIN": "exportaciones_industria", "XRM": "exportaciones_resto",
    "IBS": "importaciones", "IBI": "importaciones_bienes", "ISE": "importaciones_servicios",
    "IAP": "importaciones_agropecuario_silvicola_pesca", "IMI": "importaciones_mineria",
    "IIN": "importaciones_industria",
    "INR": "ingreso_nacional_bruto_disponible", "IKF": "tasa_inversion_capital_fijo",
}

REGION = {
    "01": "tarapaca", "02": "antofagasta", "03": "atacama", "04": "coquimbo", "05": "valparaiso",
    "06": "ohiggins", "07": "maule", "08": "biobio", "09": "araucania", "10": "los_lagos",
    "11": "aysen", "12": "magallanes", "13": "metropolitana", "14": "los_rios",
    "15": "arica_parinacota", "16": "nuble", "99": "extrarregional", "Z": "total",
}

# Dimensiones que identifican un cuadro (la "tupla" del paper). Las de serie añaden la categoría.
DIMS_CUADRO = ["indicador", "presentacion", "valoracion", "referencia", "empalme", "info_historica",
               "estacionalidad", "frecuencia", "apertura", "region_fija", "vigencia"]
DIMS_SERIE = DIMS_CUADRO + ["actividad", "region", "gasto"]


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()


def lookup_actividad(codes: tuple[str, ...], unmapped: Counter) -> str:
    codes = tuple(c for c in codes if c is not None)
    for k in range(len(codes), 0, -1):
        key = tuple(codes[:k])
        # los sufijos "Z"/"21" no aportan: probar sin ellos
        key_clean = tuple(c for c in key if c not in ("Z", "21")) or ("Z",)
        for cand in (key, key_clean):
            if cand in ACTIVIDAD:
                return ACTIVIDAD[cand]
    unmapped[".".join(codes)] += 1
    return "cod_" + "_".join(codes)


def decode(code: str, unmapped: Counter) -> dict:
    seg = code.split(".")
    fam = seg[0]
    spec = FAMILIAS[fam]
    if len(seg) != spec["n"]:
        raise ValueError(f"{code}: {len(seg)} segmentos, se esperaban {spec['n']} para {fam}")

    ref, empalme, hist = decode_hist(seg[spec["hist"]])
    val_code = seg[spec["val"]]
    if val_code == "N":
        valoracion = "corrientes"
    elif val_code == "R":
        valoracion = "encadenado" if ref >= 2008 else "constantes"
    else:
        valoracion = "na"

    row = dict(
        familia=fam,
        presentacion=PRESENTACION[seg[spec["tipo"]]],
        valoracion=valoracion,
        referencia=ref,
        empalme=empalme,
        info_historica=hist,
        estacionalidad=ESTACIONALIDAD[seg[spec["sa"]]],
        frecuencia=FRECUENCIA[seg[spec["freq"]]],
        actividad=None, region=None, gasto=None,
    )

    if fam == "F033":
        row["indicador"] = "pib"
        row["gasto"] = GASTO.get(seg[1])
        if row["gasto"] is None:
            unmapped["F033." + seg[1]] += 1
            row["gasto"] = "cod_" + seg[1]
        return row

    ind = seg[spec["ind"]]
    if ind in ("IMC", "ICF"):
        row["indicador"] = "imacec"
    elif ind == "PIB" or ind in FILAS_ESPECIALES:
        row["indicador"] = "pib"
    else:
        raise ValueError(f"{code}: indicador desconocido {ind}")

    a, b, c = spec["act"]
    act_codes = (seg[a], seg[b], seg[c] if c is not None else None)
    if ind in FILAS_ESPECIALES:
        row["actividad"] = FILAS_ESPECIALES[ind]
    else:
        row["actividad"] = lookup_actividad(act_codes, unmapped)

    if spec["region"] is not None:
        rc = seg[spec["region"]]
        row["region"] = REGION.get(rc)
        if row["region"] is None:
            unmapped["F035.region." + rc] += 1
            row["region"] = "cod_" + rc
    return row


def build(series_xlsx: Path, facetas_xlsx: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    src = pd.read_excel(series_xlsx, sheet_name="Acotacion")
    fac = pd.read_excel(facetas_xlsx, sheet_name="Cuadros_facetas")
    fac["_key"] = fac["CUADRO"].map(norm_name)
    fac = fac.drop_duplicates("_key").set_index("_key")

    unmapped: Counter = Counter()
    rows = []
    cuadro_ids: dict[str, str] = {}
    for _, r in src.iterrows():
        code = str(r["CODIGO_SERIE"]).strip()
        cuadro = re.sub(r"\s+", " ", str(r["NOMBRE CUADRO"])).strip()
        key = norm_name(cuadro)
        if key not in cuadro_ids:
            cuadro_ids[key] = f"Q{len(cuadro_ids) + 1:03d}"
        d = decode(code, unmapped)
        f = fac.loc[key] if key in fac.index else None
        rows.append(dict(
            serie_id=code,
            cuadro_id=cuadro_ids[key],
            cuadro=cuadro,
            serie=re.sub(r"\s+", " ", str(r["NOMBRE_SERIE"])).strip(),
            **d,
            apertura=(f["APERTURA"] if f is not None else None),
            vigencia={"Vigente": "vigente", "Histórico": "historico"}[str(r["VIGENCIA"]).strip()],
            unidad=str(r["UNIDAD"]).strip(),
            fecha_primer_valor=str(r["FECHA_PRIMER_VALOR"])[:10],
            _facet_estac=(f["ESTACIONALIDAD"] if f is not None else None),
            _facet_val=(f["VALORACION"] if f is not None else None),
            _facet_hist=(f["HISTORIA"] if f is not None else None),
            _facet_region=(f["REGION"] if f is not None else None),
            _facet_pres=(f["PRESENTACION"] if f is not None else None),
        ))
    df = pd.DataFrame(rows)

    # Apertura: derivada de las series (qué varía dentro del cuadro), validada contra la faceta
    def apertura_derivada(g: pd.DataFrame) -> str:
        if g["gasto"].notna().any():
            return "gasto"
        # región fija => el cuadro se abre por actividad; región variable => se abre por región
        if g["region"].nunique(dropna=True) > 1:
            return "region"
        if g["actividad"].nunique(dropna=True) > 1:
            return "actividad"
        return "total"

    def region_fija(g: pd.DataFrame):
        vals = set(g["region"].dropna()) - {"total"}
        return vals.pop() if len(vals) == 1 and g["region"].nunique(dropna=True) <= 2 else None

    ap = df.groupby("cuadro_id").apply(apertura_derivada, include_groups=False).rename("apertura_derivada")
    rf = df.groupby("cuadro_id").apply(region_fija, include_groups=False).rename("region_fija")
    df = df.merge(ap, left_on="cuadro_id", right_index=True).merge(rf, left_on="cuadro_id", right_index=True)
    df = df.rename(columns={"apertura": "apertura_faceta"})
    df["apertura"] = df["apertura_derivada"]
    # categoría: el valor de la serie sobre el eje de apertura del cuadro
    df["categoria"] = [
        r.gasto if r.apertura == "gasto" else (r.region if r.apertura == "region" else r.actividad)
        for r in df.itertuples()
    ]

    cols = ["serie_id", "cuadro_id", "cuadro", "serie", "familia"] + DIMS_SERIE + \
           ["categoria", "unidad", "fecha_primer_valor"]
    df[cols].to_csv(out / "series.csv", index=False)

    # ---- cuadros.csv: tupla agregada ------------------------------------------------
    def agg_set(s: pd.Series) -> str:
        return "|".join(sorted(str(v) for v in set(s.dropna())))

    cu = df.groupby("cuadro_id").agg(
        cuadro=("cuadro", "first"), familia=("familia", "first"), indicador=("indicador", agg_set),
        presentacion=("presentacion", agg_set), valoracion=("valoracion", agg_set),
        referencia=("referencia", agg_set), empalme=("empalme", agg_set), info_historica=("info_historica", agg_set),
        estacionalidad=("estacionalidad", agg_set), frecuencia=("frecuencia", agg_set),
        apertura=("apertura", "first"), apertura_faceta=("apertura_faceta", "first"),
        region_fija=("region_fija", "first"), vigencia=("vigencia", "first"),
        n_series=("serie_id", "size"), categorias=("categoria", agg_set),
    ).reset_index()
    cu.to_csv(out / "cuadros.csv", index=False)

    # ---- vocabularios.json ---------------------------------------------------------
    vocab = {d: dict(Counter(df[d].dropna().astype(str))) for d in DIMS_SERIE + ["categoria"]}
    # Vocabulario del modelo: sólo dimensiones lingüísticas, sólo vigentes, sin valores marginales.
    # ACTIVIDAD/REGION/GASTO se etiquetan como span genérico; sus valores son metas del normalizador.
    from catalog import DIMS_LINGUISTICAS, VALORES_MARGINALES
    vig = df[df.vigencia == "vigente"]
    vocab["modelo"] = {
        "entidades": {d.upper(): sorted(v for v in set(vig[d].dropna().astype(str)) - set(map(str, VALORES_MARGINALES.get(d, ()))))
                      for d in DIMS_LINGUISTICAS},
        "referencia_si_se_menciona": sorted(int(v) for v in set(df.referencia)),
        "cabezas_binarias": ["activ_cls", "reg_cls", "gast_cls"],
        "normalizador_restringido_por_candidatos": ["ACTIVIDAD", "REGION", "GASTO"],
    }
    vocab["_codigos"] = {
        "actividad": {".".join(k): v for k, v in ACTIVIDAD.items()},
        "filas_especiales": FILAS_ESPECIALES, "gasto": GASTO, "region": REGION,
        "presentacion": PRESENTACION, "estacionalidad": ESTACIONALIDAD, "frecuencia": FRECUENCIA,
    }
    (out / "vocabularios.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- reporte.md ----------------------------------------------------------------
    L = ["# Reporte de construcción del catálogo", ""]
    L += [f"- Series: {len(df)}", f"- Cuadros: {df.cuadro_id.nunique()}",
          f"- Cuadros sin faceta (join por nombre): {int(df.groupby('cuadro_id')._facet_estac.first().isna().sum())}",
          f"- Códigos sin mapear: {sum(unmapped.values())}", ""]
    if unmapped:
        L += ["## Códigos sin mapear", ""] + [f"- `{k}`: {v}" for k, v in unmapped.most_common()] + [""]

    L += ["## Distribución por dimensión (series)", ""]
    for d in DIMS_SERIE:
        L.append(f"- **{d}**: " + ", ".join(f"{k}={v}" for k, v in Counter(df[d].astype(str)).most_common()))
    L.append("")

    L += ["## Cuadros por vigencia y apertura", "", pd.crosstab(cu.apertura, cu.vigencia).to_markdown(), ""]

    # colisiones: cuadros con la misma tupla
    key = cu[DIMS_CUADRO].fillna("").astype(str).agg("|".join, axis=1)
    dup = cu[key.duplicated(keep=False)].assign(_k=key[key.duplicated(keep=False)]).sort_values("_k")
    L += [f"## Colisiones de tupla entre cuadros: {dup._k.nunique()} grupos, {len(dup)} cuadros", ""]
    for k, g in dup.groupby("_k"):
        L.append(f"- `{k}`")
        for r in g.itertuples():
            L.append(f"    - {r.cuadro_id} ({r.n_series} series, freq={r.frecuencia}): {r.cuadro}")
    L.append("")
    L += [f"- Tuplas de cuadro distintas: {key.nunique()} de {len(cu)} cuadros",
          f"- Producto cartesiano de los vocabularios de cuadro: "
          f"{int(pd.Series([cu[d].nunique() for d in DIMS_CUADRO]).prod()):,}", ""]

    # validación contra facetas
    L += ["## Discrepancias con las facetas de cuadro", ""]
    chk = df.dropna(subset=["_facet_estac"])
    g = chk.groupby("cuadro_id")
    est_mis = g.apply(lambda x: (x._facet_estac.iloc[0] == "desestacionalizada") != (x.estacionalidad == "desestacionalizada").all(), include_groups=False)
    L.append(f"- Estacionalidad (faceta vs. todas las series del cuadro): {int(est_mis.sum())} cuadros difieren "
             f"(esperado: cuadros mixtos como 'Producto interno bruto, referencia 2018')")
    for cid in est_mis[est_mis].index[:10]:
        x = chk[chk.cuadro_id == cid]
        L.append(f"    - {cid}: faceta={x._facet_estac.iloc[0]}, series={dict(Counter(x.estacionalidad))} — {x.cuadro.iloc[0][:90]}")
    reg_mis = g.apply(lambda x: (x._facet_region.iloc[0] not in ("na", None)) and (x.region_fija.iloc[0] != x._facet_region.iloc[0]), include_groups=False)
    L.append(f"- Región fija (faceta vs. código): {int(reg_mis.sum())} cuadros difieren")
    for cid in reg_mis[reg_mis].index[:10]:
        x = chk[chk.cuadro_id == cid]
        L.append(f"    - {cid}: faceta={x._facet_region.iloc[0]}, código={x.region_fija.iloc[0]} — {x.cuadro.iloc[0][:90]}")
    hist_mis = g.apply(lambda x: str(x.referencia.iloc[0]) not in str(x._facet_hist.iloc[0]), include_groups=False)
    L.append(f"- Referencia (faceta HISTORIA vs. código): {int(hist_mis.sum())} cuadros difieren")
    for cid in hist_mis[hist_mis].index[:10]:
        x = chk[chk.cuadro_id == cid]
        L.append(f"    - {cid}: faceta={x._facet_hist.iloc[0]}, código={x.referencia.iloc[0]} — {x.cuadro.iloc[0][:90]}")
    pres_map = {"nivel": {"nivel"}, "contribucion": {"contribucion_aa", "contribucion_pa"}, "participacion": {"participacion"}}
    pres_mis = g.apply(lambda x: not set(x.presentacion) <= pres_map.get(x._facet_pres.iloc[0], set()), include_groups=False)
    L.append(f"- Presentación (faceta vs. código): {int(pres_mis.sum())} cuadros difieren")
    for cid in pres_mis[pres_mis].index[:10]:
        x = chk[chk.cuadro_id == cid]
        L.append(f"    - {cid}: faceta={x._facet_pres.iloc[0]}, series={dict(Counter(x.presentacion))} — {x.cuadro.iloc[0][:90]}")
    ap_mis = df.groupby("cuadro_id").apply(lambda x: x.apertura_faceta.iloc[0] != x.apertura.iloc[0], include_groups=False)
    L.append(f"- Apertura (faceta vs. derivada de las series; se usa la derivada): {int(ap_mis.sum())} cuadros difieren")
    for cid in ap_mis[ap_mis].index[:40]:
        x = df[df.cuadro_id == cid]
        L.append(f"    - {cid}: faceta={x.apertura_faceta.iloc[0]}, derivada={x.apertura.iloc[0]}, n={len(x)}, categorias={sorted(set(x.categoria.dropna().astype(str)))[:6]} — {x.cuadro.iloc[0][:80]}")
    L.append("")

    # ejemplos del paper: propagación de restricciones
    L += ["## Propagación de restricciones (ejemplos del paper, sólo vigentes)", ""]
    vig = df[df.vigencia == "vigente"]
    for desc, q in [
        ("frecuencia=trimestral", dict(frecuencia="trimestral")),
        ("presentacion=contribucion_pa", dict(presentacion="contribucion_pa")),
        ("indicador=imacec, presentacion=nivel", dict(indicador="imacec", presentacion="nivel")),
        ("frecuencia=mensual", dict(frecuencia="mensual")),
    ]:
        sub = vig
        for k, v in q.items():
            sub = sub[sub[k] == v]
        collapsed = {d: (None if pd.isna(sub[d].iloc[0]) else sub[d].iloc[0].item() if hasattr(sub[d].iloc[0], "item") else sub[d].iloc[0])
                     for d in DIMS_CUADRO if sub[d].nunique(dropna=False) == 1 and d not in q}
        L.append(f"- `{desc}` → {sub.cuadro_id.nunique()} cuadros, {len(sub)} series; dimensiones que colapsan: {collapsed}")
    L.append("")
    (out / "reporte.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series-xlsx", type=Path, default=Path("data/catalogo/raw/catalogo_series_bde_2026-08-05.xlsx"))
    p.add_argument("--facetas-xlsx", type=Path, default=Path("data/catalogo/raw/cuadros_facetas_bde.xlsx"))
    p.add_argument("--out", type=Path, default=Path("data/catalogo"))
    a = p.parse_args()
    build(a.series_xlsx, a.facetas_xlsx, a.out)
