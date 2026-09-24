# Migración v3 → v4

## train: 5717 filas (471 aumentadas)

- cabezas: {'activ_cls=0': 4048, 'activ_cls=1': 1669, 'calculo=nivel': 2758, 'calculo=var_anual': 1491, 'calculo=var_periodo_anterior': 1468, 'gast_cls=0': 5234, 'gast_cls=1': 483, 'reg_cls=0': 4300, 'reg_cls=1': 1417}
- spans por entidad: {'INDICADOR': 5717, 'PERIODO': 3843, 'ESTACIONALIDAD': 3412, 'FRECUENCIA': 2702, 'ACTIVIDAD': 1613, 'REGION': 1361, 'PRESENTACION': 1294, 'GASTO': 477, 'VALORACION': 314, 'REFERENCIA': 157}
- avisos: {'contribution_con_span': 1173, 'region_specific_sin_span': 49}

## dev: 760 filas (61 aumentadas)

- cabezas: {'activ_cls=0': 552, 'activ_cls=1': 208, 'calculo=nivel': 393, 'calculo=var_anual': 184, 'calculo=var_periodo_anterior': 183, 'gast_cls=0': 706, 'gast_cls=1': 54, 'reg_cls=0': 584, 'reg_cls=1': 176}
- spans por entidad: {'INDICADOR': 760, 'PERIODO': 492, 'ESTACIONALIDAD': 470, 'FRECUENCIA': 360, 'ACTIVIDAD': 231, 'REGION': 194, 'PRESENTACION': 193, 'GASTO': 76, 'VALORACION': 41, 'REFERENCIA': 20}
- avisos: {'contribution_con_span': 183, 'region_specific_sin_span': 11}

## test: 1141 filas (93 aumentadas)

- cabezas: {'activ_cls=0': 815, 'activ_cls=1': 326, 'calculo=nivel': 551, 'calculo=var_anual': 286, 'calculo=var_periodo_anterior': 304, 'gast_cls=0': 1052, 'gast_cls=1': 89, 'reg_cls=0': 825, 'reg_cls=1': 316}
- spans por entidad: {'INDICADOR': 1141, 'PERIODO': 757, 'ESTACIONALIDAD': 686, 'FRECUENCIA': 524, 'ACTIVIDAD': 332, 'REGION': 267, 'PRESENTACION': 261, 'GASTO': 98, 'VALORACION': 62, 'REFERENCIA': 31}
- avisos: {'contribution_con_span': 232, 'region_specific_sin_span': 13}

## Filas para revisar a mano: 73

- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pib, sin efecto estacional el resultado publicado recientemente
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pib regional el último dato disponible
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pibr el resultado publicado recientemente
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pibr, sin ajustar por estacionalidad el T1 2019
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pib el segundo trimestre 2018
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pibr el T3 2019
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pib regional el T4 2018 trimestral
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pib el T3 2018
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pibr el 2do trimestre 2024
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pib el 2022-T3 periodicidad trimestral
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pib, sin ajustar por estacionalidad el 3er trimestre 2023
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pibr el del 2do trimestre 2023 al 2do trimestre 2024
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pib regional del 2do trimestre 2021 y 4to trimestre 2025 por trimestre
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pib regional el en el 3er trimestre 2026
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el producto interno bruto el a lo largo del cuarto trimestre 2026
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pibr el resultado publicado recientemente
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el producto interno bruto la última información disponible
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades incidieron mas recientemente en el pib regional datos brutos
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pib regional la últimas cifras publicadas
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pib el 2018 anual
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pib regional el 2026 periodicidad anual
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pib el 2026 anualmente
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pib regional desde 2015 hasta 2015 periodicidad anual
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pibr desde 2018 hasta 2023 anual
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el producto interno bruto de 2015 a 2025 anual
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pibr de 2023 a 2023 frecuencia anual
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pib entre 2023 y 2023 anualmente
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el producto interno bruto corregido por estacionalidad el valor más reciente publicado
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pibr, ajuste estacional la última publicación
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades incidieron mas recientemente en el pibr con ajustado por estacionalidad
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron la ultima lectura del pib regional, desestacionalizado
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pib regional con tratamiento estacional el 3er trim 2026
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pib, libre de estacionalidad el 2020-T3
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el producto interno bruto descontando el ajuste estacional desde 3er trimestre 2022 y 4to trimestre 2024
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pib regional, quitando el ajuste estacional desde 2do trimestre 2019 y 4to trimestre 2026 frecuencia trimestral
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pib regional, serie ajustada el en el 3er trimestre 2024
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pib regional, corregido por estacionalidad la información más actual
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pib regional ajuste de estacionalidad la información más actual
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades incidieron mas recientemente en el pib regional con ajustado por estacionalidad
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pib sin estacionalidad la cifra más reciente publicada
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pib regional ajuste estacional el último dato disponible
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pib regional en su versión ajustado estacionalmente el 2024 periodicidad anual
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pib, descontando el ajuste estacional el 2025 periodicidad anual
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pibr corregido por estacionalidad el 2020 frecuencia anual
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pib, serie desestacionalizada el 2018 frecuencia anual
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pibr ajuste de estacionalidad desde 2022 hasta 2023 anualmente
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pibr corregido por estacionalidad de 2015 a 2024 periodicidad anual
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pibr, eliminando el componente estacional de 2021 a 2025 frecuencia anual
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pib, con tratamiento estacional entre 2016 y 2019 periodicidad anual
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pibr el 2023-T2 frecuencia trimestral
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pibr el 2do trimestre 2023
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pib no ajustado por estacionalidad el a lo largo del 1er trimestre 2018
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pib regional, no ajustado por estacionalidad desde 3er trimestre 2018 y 3er trimestre 2019
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el producto interno bruto desde 3er trimestre 2024 y 4to trimestre 2025
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pib los datos más actualizados
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pibr, datos brutos el 2023 anualmente
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pib regional desde 2017 hasta 2025 frecuencia anual
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pib, libre de estacionalidad el T2 2025
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el producto interno bruto con ajustado estacionalmente la cifra preliminar publicada
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pib regional, con tratamiento estacional el 2024 frecuencia anual
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pib regional el 1er trimestre 2021
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pibr entre 3er trimestre 2015 y 3er trimestre 2019
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pib regional, serie original la últimas cifras publicadas
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del producto interno bruto, sin efecto estacional la información más actual
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del producto interno bruto de 2021 a 2025 anualmente
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades incidieron mas recientemente en el pib con serie desestacionalizada
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pib serie con ajuste estacional el 3er trimestre 2019 frecuencia trimestral
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pibr, serie ajustada el 3er trimestre 2020
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el producto interno bruto con ajustado por estacionalidad el T1 2022
- [region=specific sin span REGION (queda reg_cls=0)] Qué sectores explicaron mayor parte del pib, con ajuste estacional la información más actual
- [region=specific sin span REGION (queda reg_cls=0)] Qué actividades impulsaron el pib, con ajuste estacional el 2020 anualmente
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el producto interno bruto descontando el ajuste estacional desde 2016 hasta 2017 anual
- [region=specific sin span REGION (queda reg_cls=0)] Cuales actividades tuvieron mayor incidencia en el pib, ajuste estacional durante 2023 y 2025 frecuencia anual

- processed/: 7618 filas; raw/dataset.xlsx con columnas ['Utterance', 'activ_cls', 'reg_cls', 'gast_cls', 'calculo', 'NER_BIO', 'Split', 'Aumentada', 'Revisar']
