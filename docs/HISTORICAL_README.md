# Historical encuestador README

> Preserved from the repository front door during the 2026 architecture revival. This document is historical evidence. Commands, cron instructions, model claims, local paths, price-series descriptions, sampling language and fitted artifacts below are **not** current governed interfaces unless explicitly re-established elsewhere.

# encuestador-de-hogares

Entrenador de modelos de random forest que predicen respuestas a las preguntas de la encuesta permanente de hogares (EPH - INDEC).

This repository contains code and data for analyzing household survey data from the Encuesta Permanente de Hogares (EPH) in Argentina. The goal of this project is to train machine learning models to predict various household characteristics from the EPH data and to extract information from census data that can be used to improve the accuracy of these predictions. The repository includes Jupyter notebooks that load and format the EPH and census data, train machine learning models, and extract samples of data from the census. The EPHARG_train files are the training sets, the CLF files are the machine learning models saved, and the data folder contains information that is used in the analysis. The repository also includes figures that show the results of the analysis.

## Modelos

En este repositorio se utilizan modelos de Random Forest para predecir diferentes características de hogares a partir de la encuesta EPH. El objetivo es mejorar la precisión de estas predicciones al usar información del Censo Nacional. Los modelos se entrenan en cuatro etapas: clasificación 1, clasificación 2, clasificación 3, y regresión.

En la primera etapa de clasificación, se utilizan variables específicas de la EPH para predecir variables categóricas ausentes en el Censo 2010. En la segunda etapa, se utilizan las variables de la primera etapa junto con tres variables adicionales para predecir otra tanda de variables. En la tercer etapa, se utilizan las variables de las primeras dos etapas para predecir una serie de variables relacionadas con la informalidad laboral. Finalmente, en la etapa de regresión, se utilizan las variables de las tres etapas anteriores para predecir los ingresos de las personas.

Los ingresos se deflactan a valores de enero 2016 utilizando la metodología de Favata Zack Steingart, de promedio de índices provinciales. Los datos de este índice disponible el el repositorio IPC Argentina. Se transforman a logaritmos para una mejor distribución. Al aplicar estas etapas de predicción en una persona censada, se puede obtener una estimación detallada de sus ingresos y entender mejor las condiciones de vida de la población argentina.

## Actualizaciones Periodicas

Tanto los datasets de entrenamiento, como los archivos que guardan los modelos, son de tamaño demasiado grande para sincronizarse en el repositorio. Por eso se recomendaba a cada usuario clonar el repositorio y correr las rutinas `crear_EPH_training` y `entrenar_modelos`.

Históricamente se documentaron ejecuciones periódicas mediante cron para incorporar nuevos microdatos EPH. Esa automatización fue retirada durante la reactivación 2026 y no es parte de la superficie actual.

## Ejemplos históricos

El proyecto exploró heterogeneidad geográfica, deciles de ingreso, ingresos según edad, condición de actividad, aglomerados, nacionalidad y educación a partir de predicciones aplicadas sobre muestras censales.

Las figuras versionadas en `figuras/` y notebooks históricas conservan esa evidencia exploratoria.

## Metodología histórica

### Índice de precios

El proyecto aplicaba una serie de precios compuesta para expresar ingresos en una referencia monetaria común y luego transformarlos a logaritmos. La dependencia histórica fue mutable y la arquitectura moderna exige un `IPC-Argentina` monetary-conversion release exacto antes de cualquier nueva inferencia.

### Muestreos de Censo

El sistema histórico tomaba muestras de hogares del Censo 2010 y ajustaba la fracción por departamento usando proyecciones de población. Esa capacidad fue separada y actualmente pertenece a `samplerCensoARG`.

La motivación de samplear hogares en vez de personas se preserva: pobreza e indigencia se evalúan sobre hogares y el downstream necesita conservar composición y pertenencia de sus miembros.

### Actualización trimestral de empleo

Una notebook histórica de predicción modificaba `CONDACT` antes del primer clasificador para reproducir una razón trimestral de desempleo. El mecanismo no está aprobado actualmente, pero conserva una intuición importante: una variable semánticamente presente en Censo 2010 no necesariamente es una observación válida del estado de esa persona para un período de bienestar posterior.

La arquitectura moderna trata cualquier operación semejante como **target-period transport/calibration**, explícita, versionada y diagnosticada; nunca como una mutación silenciosa del dato censal.

## Estado moderno

La autoridad vigente está documentada en:

- `SYSTEM.yaml`;
- `LIFECYCLE.md`;
- `docs/FUNCTIONAL_CONTRACT.md`;
- `contracts/functional_interface.yaml`;
- `contracts/deployment_dag.yaml`.

El código, notebooks, modelos serializados, datasets y figuras históricas conservados en este repositorio son evidencia para reconstrucción, comparación y tests. No son automáticamente releases científicos actuales.
