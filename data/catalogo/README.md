# Catálogo de resolución (PIB e Imacec, BDE)

La tupla de cada serie se **deriva de su código BDE**; no se construye a mano.
`util/build_catalog.py` decodifica los 4.490 códigos del universo acotado, valida contra
las facetas por cuadro y escribe estos archivos. `catalog.py` (raíz del repo) los carga
y filtra sin dependencias externas.

```bash
.venv/bin/python util/build_catalog.py        # regenera series.csv, cuadros.csv, vocabularios.json, reporte.md
.venv/bin/python -m pytest tests/test_catalog.py
```

## Fuentes (`raw/`)

| Archivo | Hoja | Uso |
|---|---|---|
| `catalogo_series_bde_2026-08-05.xlsx` | `Acotacion` | Universo: 4.490 series, 276 cuadros, con vigencia, unidad y frecuencia |
| `cuadros_facetas_bde.xlsx` | `Cuadros_facetas` | Facetas por cuadro (297). Sólo se usa para validar; la apertura se deriva de las series |

## Estructura de los códigos de serie

```
F032 (10 seg)  F032.{ind}.{tipo}.{val}.{mon}.{hist}.{act}.{subact}.{sa}.{freq}          PIB por actividad, Imacec
F033 ( 8 seg)  F033.{gasto}.{tipo}.{val}.{mon}.{hist}.{sa}.{freq}                       Gasto del PIB
F035 (12 seg)  F035.{ind}.{tipo}.{val}.{mon}.{hist}.{act}.{subact}.{sub2}.{region}.{sa}.{freq}   PIB regional
```

| Segmento | Valores | Dimensión |
|---|---|---|
| `ind` | `PIB`, `IMC`/`ICF` (Imacec), `IMB`/`IMP`/`PCF` (filas especiales regionales) | `indicador` |
| `tipo` | `FLU`/`IND` nivel, `V12` contribución a/a, `VAR` contribución período anterior, `RPA` participación, `TAS` tasa | `presentacion` |
| `val` | `N` corrientes, `R` real (encadenado si referencia ≥ 2008, constantes si anterior), `Z` no aplica | `valoracion` |
| `hist` | año (`1977`…`2018`), `EPxx` empalmada, `HIST`/`HIST13` información histórica | `referencia`, `empalme`, `info_historica` |
| `act`, `subact`, `sub2` | códigos de actividad (ver `vocabularios.json` → `_codigos.actividad`) | `actividad` |
| `gasto` | componente del gasto (`DDI`, `CTO`, `FKF`, `XBS`, …) | `gasto` |
| `region` | `01`…`16` regiones, `99` extrarregional, `Z` total nacional | `region` |
| `sa` | `0` original, `1`/`3` desestacionalizada, `4` tendencia-ciclo | `estacionalidad` |
| `freq` | `M`, `T`, `A` | `frecuencia` |

## Archivos generados

- **`series.csv`**: una fila por par cuadro–serie. Columnas de identidad de cuadro
  (`indicador, presentacion, valoracion, referencia, empalme, info_historica, estacionalidad,
  frecuencia, apertura, region_fija, vigencia`) más la categoría de la serie
  (`actividad`, `region`, `gasto`; `categoria` = la del eje de apertura).
- **`cuadros.csv`**: una fila por cuadro con la tupla agregada (valores múltiples separados por `|`).
- **`vocabularios.json`**: valores canónicos por dimensión con conteos, y las tablas código → slug.
- **`reporte.md`**: colisiones de tupla, códigos sin mapear, discrepancias con las facetas y
  ejemplos de propagación de restricciones.

## Tres clases de dimensión

| Clase | Dimensiones | Quién la fija |
|---|---|---|
| Lingüísticas (8) | indicador, frecuencia, estacionalidad, presentacion, valoracion, actividad, region, gasto | el extractor (NER + normalización, o cabeza) |
| Rara (1) | referencia | mención excepcional ("base 2013"); si no, 2018 |
| Estructurales (5) | vigencia, empalme, info_historica, apertura, region_fija | preferencias del resolvedor o derivadas del catálogo |

`vocabularios.json["modelo"]` contiene sólo las lingüísticas, sobre cuadros vigentes y sin los
valores marginales (`tasa`, `tendencia_ciclo`, `extrarregional`). ACTIVIDAD, REGION y GASTO se
etiquetan como span genérico; sus valores son metas del normalizador restringido a los candidatos.

## Resolvedor (`catalog.py`)

```python
from catalog import Catalog
r = Catalog.load().resolve(region="valparaiso", actividad="comercio")
r.estado       # "resuelto" | "aclarar" | "no_existe"
r.cuadros      # {'Q029': 'PIB actividad económica, Región de Valparaíso, ...'}
r.supuestos    # {'presentacion': 'nivel', 'valoracion': 'encadenado'}  <- se explicitan al usuario
r.colapsadas   # {'indicador': 'pib', 'estacionalidad': 'original', ...} <- propagación de restricciones
r.aclarar      # ('estacionalidad', Counter({...})) cuando el estado es "aclarar"
```

`resolve` trabaja sobre `espacio_trabajo()` (vigentes, sin marginales) salvo que la consulta nombre
una referencia distinta de 2018, en cuyo caso abre el universo histórico. Las `PREFERENCIAS` son
defaults blandos en orden: se imponen sólo si algún candidato las cumple. La tabla condicionada de
estacionalidad y presentación según indicador y cálculo (paper, sección 4.4) aún no está implementada;
mientras tanto esos casos terminan en pregunta aclaratoria.

`resolve_consulta(cabezas, **q)` recibe además las cabezas `activ_cls`/`reg_cls`/`gast_cls` y aplica
el Cuadro 2 del paper: sin span ni cabeza, la fila agregada; con span, las categorías mencionadas;
con cabeza, el cuadro completo (con las mencionadas en `destacadas`). Cada eje conserva su propia
cardinalidad, y los cuadros que difieren sólo en `apertura` o `region_fija` (cortes alternativos
del mismo dato) no piden aclaración.

## Convenciones

- `total` en `actividad`/`region` es la fila agregada de un cuadro abierto por ese eje; vacío
  significa que la dimensión no aplica a esa familia.
- `region_fija` identifica los cuadros de una región abiertos por actividad.
- `apertura` se deriva de qué varía entre las series del cuadro (región > actividad > gasto > total);
  la faceta original queda en `cuadros.csv` como `apertura_faceta` para revisión.
