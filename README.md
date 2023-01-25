# encuestador-de-hogares

Entrenador de modelos de random forest que predicen respuestas a las preguntas de la encuesta permanente de hogares (EPH - INDEC).

This repository contains code and data for analyzing household survey data from the Encuesta Permanente de Hogares (EPH) in Argentina. The goal of this project is to train machine learning models to predict various household characteristics from the EPH data and to extract information from census data that can be used to improve the accuracy of these predictions. The repository includes Jupyter notebooks that load and format the EPH and census data, train machine learning models, and extract samples of data from the census. The EPHARG_train files are the training sets, the CLF files are the machine learning models saved, and the data folder contains information that is used in the analysis. The repository also includes figures that show the results of the analysis.




## Modelos

En este repositorio se utilizan modelos de Random Forest para predecir diferentes características de hogares a partir de la encuesta EPH. El objetivo es mejorar la precisión de estas predicciones al usar información del Censo Nacional. Los modelos se entrenan en cuatro etapas: clasificación 1, clasificación 2, clasificación 3, y regresión.

En la primera etapa de clasificación, se utilizan variables específicas de la EPH para predecir variables categóricas ausentes en el Censo 2010. En la segunda etapa, se utilizan las variables de la primera etapa junto con tres variables adicionales para predecir otra tanda de variables. En la tercer etapa, se utilizan las variables de las primeras dos etapas para predecir una serie de variables relacionadas con la informalidad laboral. Finalmente, en la etapa de regresión, se utilizan las variables de las tres etapas anteriores para predecir los ingresos de las personas.

Los ingresos se deflactan a valores de enero 2016 utilizando la metodología de Favata Zack Steingart, de promedio de índices provinciales. Los datos de este índice disponible el el repositorio [IPC Argentina](https://github.com/matuteiglesias/IPC-Argentina). Se transforman a logaritmos para una mejor distribución. Al aplicar estas etapas de predicción en una persona censada, se puede obtener una estimación detallada de sus ingresos y entender mejor las condiciones de vida de la población argentina.


## Actualizaciones Periodicas:

Tanto los datasets de entrenamiento, como los archivos que guardan los modelos, son de tamaño demasiado grande para sincronizarse en el repositorio. Por eso se recomienda a cada usuario clonar el repositorio, y correr las rutinas 'crear_EPH_training' y 'entrenar_modelos'. 

Esto se hace simplemente abriendo una terminal, ubicandose en el directorio rutinas y corriendo:

`python crear_EPH_training.py -y 2003 2023 -ow False`

`python entrenar_modelos.py -y 2003 2023 -ow False`

Estas dos rutinas generan los datasets de entrenamiento y entrenan los modelos. Se necesitan los microdatos de EPH elaborados por INDEC, los cuales estan disponibles en el repositorio: https://github.com/matuteiglesias/microdatos-EPH-INDEC

Se puede configurar mediante crontab la ejecucion de estas rutinas todos los dias, con opcion `-ow False` de forma de incorporar datos nuevos subidos por INDEC en un plazo de no mas de 24 hs. 

El INDEC se toma al menos 130 dias luego de terminado el trimestre para publicar los microdatos. De forma que las publicaciones de microdatos se esperan en la primera mitad de los meses de febrero, mayo, agosto y noviembre. 

## Ejemplos:

***Heterogeneidad geografica***

Ingresos de habitantes provinciales medianos (AR$ corrientes de 2021-01) y su ranking segun el ingreso individual nacional.

<img src="/figuras/plot_6.png" width="400">

Por ejemplo, el habitante medio de CABA gana casi $49000 en enero de 2021, un ingreso mayor que el 70% de los argentinos.

Dependiendo el tamano de sampleo, se puede alcanzar una resolucion geografica satisfactoria en fracciones o radios censales. El siguiente mapa repite el calculo del mapa anterior pero en las fracciones censales del AMBA.

<img src="/figuras/plot_5.png" width="400">


***Deciles de ingreso***

El encuestador permite observar la composicion de los deciles de ingreso individual. Por ejemplo en el siguiente grafico tenemos la composicion de deciles por genero, donde vemos que los deciles mas alto tienen mayoria de hombres.

![Plot](/figuras/plot_4.png)

***Ingresos segun edad***

Podemos tambien estudiar desagregados por edad. Esto se muestra en el siguiente grafico donde ademas de la evolucion de ingreso por edad tipica, se ve una brecha de genero importante. Esta brecha se cierra con la jubilacion, que en Argentina es casi universal.

![Plot](/figuras/plot_8.png)

***Ingresos segun condicion de actividad***

El mayor determinante de ingresos individuales es la condicion de actividad, es decir, si la persona esta ocupada, desocupada, o es inactiva. Los jubilados, estudiantes, amas de casa son inactivos.

![Plot](/figuras/plot_2.png)

***Aglomerados rankeados por ingreso***

En el siguiente grafico, la distribucion de ingresos individuales segun aglomerado:

![Plot](/figuras/plot_3.png)

***Extranjeros rankeados por ingreso***

Y tambien podemos consultar los ingresos segun nacionalidad de los inmigrantes y compararlos con los argentinos.

![Plot](/figuras/plot_9.png)

***Nivel educativo y nacionalidad***

Podemos ver tambien la distribucion de ingresos segun nivel educativo y nacionalidad (argentino - extranjero).

![Plot](/figuras/plot_7.png)

***Nivel educativo y genero***

Similarmente, podemos ver las brechas de genero segun nivel educativo:

![Plot](/figuras/plot_0.png)

## Metodologia:

***Indice de Precios***

El indice de precios mensual se grafica a continuacion. Para el periodo de enero de 2016 al presente la fuente es el INDEC y informacion se descarga del sistema de almacenamiento de archivos y catálogos de infra.datos.gob.ar.

Para el periodo de 2003 a 2015 se usa provisoriamente la serie ofrecida por pricestats, tambien conocida como "inflacion verdadera". El uso de esta serie puede ser desafiado y abandonado en el futuro por un indice consensuadamente mas robusto.

La serie de indice de precios en infra.datos.gob.ar suele terminar algunos meses antes del presente. El nivel de precios en el presente se extiende aplicando el ritmo promedio de aumento de los ultimos 6 meses con informacion disponible.

***Muestreos de Censo***

Se usan las proyecciones de poblacion por departamento (2001 - 2025) elaboradas por INDEC.

Se toman muestreos de la informacion censal de 2010 (40117096 filas).

Para muestreo general randomizado se elige una fraccion de muestreo (ej, 1%, 2%, 4%). Se multiplica la fraccion de muestreo por la razon entre poblacion del departamento en el año proyectado y poblacion observada en el departamento en 2010. Esta nueva fraccion nos indica cuantos hogares tomar aleatoriamente para simular un departamento en un año a eleccion. Por ejemplo, simula tomar el 1% de la poblacion del partido de Tigre en 2021. Si se repite para el resto de los departamentos para el mismo año, tenemos un sampleo teorico del 1% de hogares de Argentina en 2021.

Las proyecciones de poblacion cuentan personas pero sampleamos hogares. Las poblaciones de personas que obtenemos con este metodo por lo tanto no se ajustan con precision a las proyecciones que tomamos como base, aunque si se aproximan a ella a los fines practicos. 

El sampleo de personas no es conveniente porque la incidencia de pobreza e indigencia se calculan a nivel de hogar.


## Datos


Recordar que es posible acceder a la base de datos de más de 40 millones de filas asociadas a las personas censadas en 2010. 

Se usan las estimaciones de población por departamento (partido, comuna) de 2001 al presente publicadas por INDEC para simular los tamaños poblacionales relativos.

Se usa información de niveles de empleo nacionales para sobre-samplear (sub-samplear) personas ocupadas o desocupadas censadas en 2010 en magnitud correspondiente.


