# encuestador-de-hogares
Entrenador de modelos de random forest que predicen respuestas a las preguntas de la encuesta permanente de hogares (EPH - INDEC).

Principalmente, estos modelos permiten predecir los ingresos de las personas segun las caracteristicas capturadas en el censo. Entre las ventajas de esta herramienta podemos estudiar la dependencia de ingresos segun mas de una variable al mismo tiempo, computar muy facilmente percentiles (deciles) de ingreso, contar con una definicion geografica de radio censal, entre otras ventajas muy utiles.

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
