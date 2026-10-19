# Reporte de construcción del catálogo

- Series: 4490
- Cuadros: 276
- Cuadros sin faceta (join por nombre): 0
- Códigos sin mapear: 0

## Distribución por dimensión (series)

- **indicador**: pib=4393, imacec=97
- **presentacion**: nivel=3309, contribucion_aa=893, contribucion_pa=171, participacion=115, tasa=2
- **valoracion**: encadenado=2279, corrientes=1318, constantes=796, na=97
- **referencia**: 2018=1510, 2013=1012, 2008=980, 2003=338, 1996=312, 1986=284, 1977=54
- **empalme**: False=3913, True=577
- **info_historica**: False=4430, True=60
- **estacionalidad**: original=3800, desestacionalizada=685, tendencia_ciclo=5
- **frecuencia**: anual=2595, trimestral=1798, mensual=97
- **apertura**: actividad=3320, gasto=895, region=255, total=20
- **region_fija**: nan=2280, tarapaca=150, antofagasta=150, atacama=150, coquimbo=150, valparaiso=150, metropolitana=150, ohiggins=150, aysen=150, magallanes=150, maule=148, biobio=148, araucania=148, los_lagos=148, arica_parinacota=108, los_rios=106, nuble=54
- **vigencia**: historico=2980, vigente=1510
- **actividad**: nan=858, total=511, industria_manufacturera=224, mineria=216, agropecuario_silvicola=184, pesca=184, electricidad_gas_agua=184, construccion=184, servicios_vivienda=184, servicios_personales=184, administracion_publica=184, servicios_financieros_empresariales=156, transporte_comunicaciones=152, comercio_restaurantes_hoteles=121, comercio=113, restaurantes_hoteles=73, imputaciones_bancarias=48, produccion_bienes=40, resto_bienes=40, servicios=40, servicios_financieros=38, mineria_cobre=32, otras_actividades_mineras=32, textil_prendas_cuero=32, maderas_muebles=32, celulosa_papel_imprentas=32, minerales_no_metalicos_metalica_basica=32, productos_metalicos_maquinaria_equipo=32, transporte=32, comunicaciones=32, iva=28, derechos_importacion=28, costo_factores=26, alimentos=25, bebidas_tabaco=25, servicios_empresariales=25, quimica_petroleo_caucho_plastico=21, alimentos_bebidas_tabaco=17, refinacion_petroleo=16, quimica_caucho_plastico=16, no_minero=14, subtotal_regionalizado=14, impuestos_productos=13, iva_derechos_importacion=9, subtotal=7
- **region**: nan=2034, tarapaca=164, antofagasta=164, atacama=164, coquimbo=164, valparaiso=164, metropolitana=164, ohiggins=164, aysen=164, magallanes=164, maule=162, biobio=162, araucania=162, los_lagos=162, arica_parinacota=119, los_rios=117, nuble=59, total=26, extrarregional=11
- **gasto**: nan=3632, fbcf=50, exportaciones=50, importaciones=50, demanda_interna=46, consumo_hogares=38, consumo_gobierno=38, consumo_total=34, construccion_otras_obras=34, maquinaria_equipo=34, bienes_durables=32, bienes_no_durables=32, servicios=32, exportaciones_bienes=29, exportaciones_mineria=29, exportaciones_industria=29, exportaciones_servicios=29, importaciones_bienes=29, importaciones_agropecuario_silvicola_pesca=29, importaciones_mineria=29, importaciones_industria=29, importaciones_servicios=29, variacion_existencias=28, exportaciones_agropecuario_silvicola_pesca=25, exportaciones_cobre=18, exportaciones_resto=18, resto_demanda_interna=14, pib=7, exportaciones_agropecuario_silvicola=4, exportaciones_pesca=4, formacion_bruta_capital=2, maquinaria_equipo_nacional=2, maquinaria_equipo_importada=2, tasa_inversion_capital_fijo=2, ingreso_nacional_bruto_disponible=1

## Cuadros por vigencia y apertura

| apertura   |   historico |   vigente |
|:-----------|------------:|----------:|
| actividad  |         136 |        77 |
| gasto      |          34 |         8 |
| region     |           9 |         5 |
| total      |           6 |         1 |

## Colisiones de tupla entre cuadros: 1 grupos, 2 cuadros

- `pib|nivel|encadenado|2018|False|True|False|original|trimestral|region||vigente`
    - Q023 (19 series, freq=trimestral): Producto interno bruto trimestral por región, volumen a precios del año anterior encadenado, series empalmadas, referencia 2018 (miles de millones de pesos encadenados)
    - Q074 (19 series, freq=trimestral): Producto interno bruto anual por región, volumen a precios del año anterior encadenado, series empalmadas, referencia 2018 (miles de millones de pesos encadenados)

- Tuplas de cuadro distintas: 275 de 276 cuadros
- Producto cartesiano de los vocabularios de cuadro: 5,160,960

## Discrepancias con las facetas de cuadro

- Estacionalidad (faceta vs. todas las series del cuadro): 0 cuadros difieren (esperado: cuadros mixtos como 'Producto interno bruto, referencia 2018')
- Región fija (faceta vs. código): 8 cuadros difieren
    - Q024: faceta=arica, código=arica_parinacota — PIB por actividad económica, Región de Arica y Parinacota, volumen a precios del año anter
    - Q041: faceta=arica, código=arica_parinacota — Producto interno bruto Región de Arica y Parinacota, contribución porcentual respecto de i
    - Q058: faceta=arica, código=arica_parinacota — PIB por actividad económica, Región de Arica y Parinacota, precios corrientes, series empa
    - Q076: faceta=arica, código=arica_parinacota — PIB por actividad económica, Región de Arica y Parinacota, contribución porcentual respect
    - Q115: faceta=arica, código=arica_parinacota — PIB por actividad económica, Región de Arica y Parinacota, precios corrientes, series empa
    - Q132: faceta=arica, código=arica_parinacota — PIB por actividad económica, Región de Arica y Parinacota, volumen a precios del año anter
    - Q169: faceta=arica, código=arica_parinacota — PIB por actividad económica, XV Región de Arica y Parinacota, precios corrientes, referenc
    - Q186: faceta=arica, código=arica_parinacota — PIB por actividad económica, XV Región de Arica y Parinacota, volumen a precios del año an
- Referencia (faceta HISTORIA vs. código): 0 cuadros difieren
- Presentación (faceta vs. código): 2 cuadros difieren
    - Q275: faceta=nivel, series={'nivel': 10, 'tasa': 1} — Gasto del producto interno bruto , a precios corrientes, base 1977 (millones de pesos)
    - Q276: faceta=nivel, series={'nivel': 10, 'tasa': 1} — Gasto del producto interno bruto, a precios constantes, base 1977 (millones de pesos)
- Apertura (faceta vs. derivada de las series; se usa la derivada): 20 cuadros difieren
    - Q041: faceta=region, derivada=actividad, n=7, categorias=['comercio', 'industria_manufacturera', 'mineria', 'produccion_bienes', 'resto_bienes', 'servicios'] — Producto interno bruto Región de Arica y Parinacota, contribución porcentual res
    - Q042: faceta=region, derivada=actividad, n=7, categorias=['comercio', 'industria_manufacturera', 'mineria', 'produccion_bienes', 'resto_bienes', 'servicios'] — Producto interno bruto Región de Tarapacá, contribución porcentual respecto de i
    - Q043: faceta=region, derivada=actividad, n=7, categorias=['comercio', 'industria_manufacturera', 'mineria', 'produccion_bienes', 'resto_bienes', 'servicios'] — Producto interno bruto Región de Antofagasta, contribución porcentual respecto d
    - Q044: faceta=region, derivada=actividad, n=7, categorias=['comercio', 'industria_manufacturera', 'mineria', 'produccion_bienes', 'resto_bienes', 'servicios'] — Producto interno bruto Región de Atacama, contribución porcentual respecto de ig
    - Q045: faceta=region, derivada=actividad, n=7, categorias=['comercio', 'industria_manufacturera', 'mineria', 'produccion_bienes', 'resto_bienes', 'servicios'] — Producto interno bruto Región de Coquimbo, contribución porcentual respecto de i
    - Q046: faceta=region, derivada=actividad, n=7, categorias=['comercio', 'industria_manufacturera', 'mineria', 'produccion_bienes', 'resto_bienes', 'servicios'] — Producto interno bruto Región de Valparaíso, contribución porcentual respecto de
    - Q047: faceta=region, derivada=actividad, n=7, categorias=['comercio', 'industria_manufacturera', 'mineria', 'produccion_bienes', 'resto_bienes', 'servicios'] — Producto interno bruto Región Metropolitana de Santiago, contribución porcentual
    - Q048: faceta=region, derivada=actividad, n=7, categorias=['comercio', 'industria_manufacturera', 'mineria', 'produccion_bienes', 'resto_bienes', 'servicios'] — Producto interno bruto Región de Libertador General Bernardo O`Higgins, contribu
    - Q049: faceta=region, derivada=actividad, n=6, categorias=['comercio', 'industria_manufacturera', 'produccion_bienes', 'resto_bienes', 'servicios', 'total'] — Producto interno bruto Región del Maule, contribución porcentual respecto de igu
    - Q050: faceta=region, derivada=actividad, n=6, categorias=['comercio', 'industria_manufacturera', 'produccion_bienes', 'resto_bienes', 'servicios', 'total'] — Producto interno bruto Región de Ñuble, contribución porcentual respecto de igua
    - Q051: faceta=region, derivada=actividad, n=6, categorias=['comercio', 'industria_manufacturera', 'produccion_bienes', 'resto_bienes', 'servicios', 'total'] — Producto interno bruto Región del Biobío, contribución porcentual respecto de ig
    - Q052: faceta=region, derivada=actividad, n=6, categorias=['comercio', 'industria_manufacturera', 'produccion_bienes', 'resto_bienes', 'servicios', 'total'] — Producto interno bruto Región de La Araucanía, contribución porcentual respecto 
    - Q053: faceta=region, derivada=actividad, n=6, categorias=['comercio', 'industria_manufacturera', 'produccion_bienes', 'resto_bienes', 'servicios', 'total'] — Producto interno bruto Región de Los Ríos, contribución porcentual respecto de i
    - Q054: faceta=region, derivada=actividad, n=6, categorias=['comercio', 'industria_manufacturera', 'produccion_bienes', 'resto_bienes', 'servicios', 'total'] — Producto interno bruto Región de Los Lagos, contribución porcentual respecto de 
    - Q055: faceta=region, derivada=actividad, n=7, categorias=['comercio', 'industria_manufacturera', 'mineria', 'produccion_bienes', 'resto_bienes', 'servicios'] — Producto interno bruto Región de Aysén del General Carlos Ibáñez del Campo, cont
    - Q056: faceta=region, derivada=actividad, n=7, categorias=['comercio', 'industria_manufacturera', 'mineria', 'produccion_bienes', 'resto_bienes', 'servicios'] — Producto interno bruto Región de Magallanes y de la Antártica Chilena, contribuc
    - Q201: faceta=actividad, derivada=total, n=2, categorias=['total'] — Indicador mensual de actividad económica, Imacec, series empalmadas (índice 2008
    - Q231: faceta=actividad, derivada=total, n=3, categorias=['total'] — Indicador mensual de actividad económica, Imacec, series empalmadas (índice 2003
    - Q232: faceta=total, derivada=gasto, n=5, categorias=['ingreso_nacional_bruto_disponible'] — PIB e ingreso, series empalmadas, base 2003 (millones de pesos)
    - Q254: faceta=actividad, derivada=total, n=2, categorias=['total'] — Indicador mensual de actividad económica, Imacec (índice 1986=100)

## Propagación de restricciones (ejemplos del paper, sólo vigentes)

- `frecuencia=trimestral` → 51 cuadros, 654 series; dimensiones que colapsan: {'indicador': 'pib', 'referencia': 2018, 'info_historica': False, 'vigencia': 'vigente'}
- `presentacion=contribucion_pa` → 3 cuadros, 60 series; dimensiones que colapsan: {'referencia': 2018, 'info_historica': False, 'estacionalidad': 'desestacionalizada', 'region_fija': None, 'vigencia': 'vigente'}
- `indicador=imacec, presentacion=nivel` → 2 cuadros, 20 series; dimensiones que colapsan: {'valoracion': 'na', 'referencia': 2018, 'empalme': True, 'info_historica': False, 'frecuencia': 'mensual', 'apertura': 'actividad', 'region_fija': None, 'vigencia': 'vigente'}
- `frecuencia=mensual` → 4 cuadros, 38 series; dimensiones que colapsan: {'indicador': 'imacec', 'valoracion': 'na', 'referencia': 2018, 'info_historica': False, 'apertura': 'actividad', 'region_fija': None, 'vigencia': 'vigente'}
