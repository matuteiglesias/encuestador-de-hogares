**Variables X de la Primera etapa**

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

**Variables Y de la Primera etapa**

'CAT_OCUP': categoria de ocupado (obrero, patron, etc)

'CAT_INAC': categoría de inactividad (estudiante, jubilado, etc)

'CH07': estado civil

**Variables Y de la Segunda etapa**

'INGRESO': tiene algún ingreso?

'INGRESO_NLB' tiene ingreso no laborable?

'INGRESO_JUB': tiene ingreso jubilatorio?

'INGRESO_SBS': tiene ingreso por subsidio?

**Variables Y de la Tercera etapa**

'PP07G1': vacaciones pagas?

'PP07G_59': no tiene vacaciones, ni aguinaldo, ni días por enfermedad ni obra social

'PP07I': hace aporte jubilatorio

'PP07J': trabaja de dia/tarde/noche

'PP07K': cuando cobra le dan recibo / papel / no le dan nada

**Variables Y de la Cuarta etapa (Regresor)**

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
    
