# encuestador-de-hogares
Entrenador de modelos de random forest que predicen respuestas a las preguntas de la encuesta permanente de hogares (EPH - INDEC).

Principalmente, estos modelos permiten predecir los ingresos de las personas segun las caracteristicas capturadas en el censo. Entre las ventajas de esta herramienta podemos estudiar la dependencia de ingresos segun mas de una variable al mismo tiempo, computar muy facilmente percentiles (deciles) de ingreso, contar con una definicion geografica de radio censal, entre otras ventajas muy utiles.

Este repositorio contiene notebooks y rutinas (.py) encargadas de:

   - Crear datasets de entrenamiento a partir de microdatos de la Encuenta de Hogares de INDEC (EPH)

   - Entrenar modelo de Machine Learning en estos datos 


El modelo se llama encuestador de hogares. Cuando le suministramos informacion de personas que fueron censadas, predice variables que se preguntan en la Encuesta de Hogares. Por ejemplo, ingreso, formalidad laboral, etc.

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

## Resumen Metodologico

Los modelos usados para predecir son Random Forests en etapas. 

La idea es que si los modelos se entrenan en información de la EPH, en particular, de variables que están incluidas en el Censo Nacional, entonces luego podremos predecir información exclusiva de EPH (por ejemplo ingresos) en la población que fue censada.

Recordar que es posible acceder a la base de datos de más de 40 millones de filas asociadas a las personas censadas en 2010. 

Se usan las estimaciones de población por departamento (partido, comuna) de 2001 al presente publicadas por INDEC para simular los tamaños poblacionales relativos.

Se usa información de niveles de empleo nacionales para sobre-samplear (sub-samplear) personas ocupadas o desocupadas censadas en 2010 en magnitud correspondiente.

Se entrenan modelos para cuatro etapas:
- Clasificación 1
- Clasificación 2
- Clasificación 3
- Regresion

Las variables incluidas en cada etapa se detallan a continuación:

### Clasificación 1

En la primer etapa, se usan las variables:

'IX_TOT': tamano del hogar

'P02': sexo

'P03': edad

'AGLO rk': ranking del aglomerado por ingreso promedio individual en EPH

'Reg_rk': ranking de la región por ingreso promedio individual en EPH

'V01': tipo de vivienda particular

'H05': material de pisos

'H06': material de techo

'H07': tiene cielorraso

'H08': agua en vivienda/terreno

'H09': fuente del agua

'H10': tiene bano

'H11': baño tiene desagüe

'H12': tipo de desagüe

'H16': cuantas habitaciones

'H15': cuantos dormitorios

'PROP': régimen de propiedad

'H14': combustible de cocina

'H13': baño compartido

'P07': sabe leer

'P08': asistió al colegio

'P09': que nivel cursa o curso

'P10': nivel completo

'P05': nacionalidad argentina?

'CONDACT': condición de actividad

Todas estas variables se usan en el modelo que se entrena en EPH para predecir las siguientes variables categóricas, ausentes del Censo 2010:

'CAT_OCUP': categoria de ocupado (obrero, patron, etc)

'CAT_INAC': categoría de inactividad (estudiante, jubilado, etc)

'CH07': estado civil


### Clasificación 2

El segundo modelo de clasificación usa las 25 variables iniciales, además de las 3 variables que el primer modelo tiene como output.

Se entrena un modelo clasificador usando estas 28 variables, para predecir la siguiente tanda de variables:

'INGRESO': tiene algún ingreso?

'INGRESO_NLB' tiene ingreso no laborable?

'INGRESO_JUB': tiene ingreso jubilatorio?

'INGRESO_SBS': tiene ingreso por subsidio?

### Clasificación 3

El tercer modelo de clasificación usa las 32 variables mencionadas hasta ahora, para predecir una tanda de variables de la sección PP07 de la EPH relacionadas con la informalidad laboral:


'PP07G1': vacaciones pagas?

'PP07G_59': no tiene vacaciones, ni aguinaldo, ni días por enfermedad ni obra social

'PP07I': hace aporte jubilatorio

'PP07J': trabaja de dia/tarde/noche

'PP07K': cuando cobra le dan recibo / papel / no le dan nada


### Regresion
Finalmente, se usan las 37 variables mencionadas hasta ahora para entrenar en la EPH un modelo de regresión que predice los ingresos de las personas.

Los ingresos son deflactados a valores de enero 2016 según la metodología de Favata Zack Steingart, de promedio de índices provinciales. Los datos de este índice disponible el el repositorio [IPC Argentina](https://github.com/matuteiglesias/IPC-Argentina).

Además, se toma log10 para una adecuada distribución de las magnitudes de ingresos, que suelen ser log-normales. Se suma un mínimo para evitar el log(0).

Las variables de ingresos son:

'P21': ingreso de la ocupación principal

'P47T': ingreso total (laboral y no laboral)

'PP08D1': sueldos, jornales, etc.

'TOT_P12': ingresos por otras ocupaciones

'T_VI': total de ingresos no laborales

'V12_M': cuota de alimentos o ayuda de personas fuera del hogar

'V2_M': jubilacion o pension

'V3_M': indemnizacion por despido

'V5_M': subsidio o ayuda del gobierno, iglesias, etc.

A la información de una persona censada, se le puede aplicar las distintas etapas de predicción para terminar dando con una estimación de sus ingresos. De esta forma podemos obtener estimaciones detalladas de ingresos de subpoblaciones personalizadas.


## Mas Ejemplos:

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

***Armonizacion de respuestas de Censo 2010 a EPH***

Se necesita adaptar algunas de las categorias de respuestas para que sean equivalentes entre el Censo 2010 y la Encuesta Permanente de Hogares. En particular:

**Tipo de vivienda particular** (V01)

Respuesta Censo --> Respuesta EPH

1. Casa --> 1. Casa
2. Rancho --> 6. Otros
3. Casilla --> 6. Otros
4. Departamento --> 2. Departamento
5. Pieza en inquilinato --> 3. Pieza de inquilinato
6. Pieza en hotel familiar o pensión --> 4. Pieza en hotel/pensión
7. Local no construido para habitación --> 5. Local no construido para habitación
8. Vivienda móvil --> 6. Otros

**Material predominante de la cubierta exterior del techo** (H06)

Respuesta Censo --> Respuesta EPH
<!-- 
1. cubierta asfáltica o membrana -> 1. Membrana/cubierta asfáltica
2. baldosa o losa (sin cubierta) -> 2. Baldosa/losa sin cubierta
3. pizarra o teja -> 3. Pizarra/teja
4. chapa de metal (sin cubierta) -> 4. Chapa de metal sin cubierta
5. chapa de fibrocemento o plástico -> 5. Chapa de fibrocemento/plástico
6. chapa de cartón -> 6. Chapa de cartón
7. caña, palma, tabla o paja con o sin barro -> 7. Caña/tabla/paja con barro/paja sola 
-->

9. Otro --> 9. N/S. Depto en propiedad horizontal

**El agua que usa, ¿proviene de...** (H09)

Respuesta Censo --> Respuesta EPH

4. pozo?  --> 4. Otra fuente
5. transporte por cisterna? --> 4. Otra fuente
6. agua de lluvia, río, canal, arroyo o acequia? --> 4. Otra fuente

**Y en total, ¿cuántas habitaciones o piezas tiene este hogar? (sin contar baño/s y cocina/s)** (H16)

Esta respuesta tiene que clippearse en 9. Es decir, la EPH no acepta valores de mas de 9 para esta pregunta.


**Para cocinar, ¿utiliza principalmente...** (H14)

Respuesta Censo --> Respuesta EPH

2. gas a granel (zeppelin)? --> 4. Otro
3. gas en tubo? --> 2. Gas de tubo/garrafa
4. gas en garrafa? --> 4. Otro
5. electricidad? --> 4. Otro
6. leña o carbón? --> 3. Kerosene/ leña/ carbón
7. Otro --> 4. Otro
8 --> 9

<!-- 
**El baño / letrina, ¿es...** (H13)
4. gas en garrafa?  4. Otro
    table['H13'] = table['H13'].map({1:1, 2:2, 4:0})
.... hay algo raro...
     -->  
    
