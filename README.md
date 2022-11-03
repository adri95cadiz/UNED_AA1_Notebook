Notebook de Jupyter de mis ejercicios en la asignatura Aprendizaje Automático 1 de la UNED.

Cada uno de los notebooks se refiere a un capítulo o a sus actividades del libro Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow Second Edition de O'Reilly.

Los notebooks contienen las explicaciones y pruebas del libro realizadas, así cómo mis ejercicios y prácticas en la asignatura. 

Se puede encontrar el código fuente original del libro en el este repositorio: https://github.com/ageron/handson-ml2

# Machine Learning Checklist.

He traducido la check-list que se encuentra en el repositorio del libro: https://github.com/adri95cadiz/handson-ml2/blob/master/ml-project-checklist.md para referencia
propia o para quien pueda interesar. 

Esta checklist puede servir de guía en tus proyectos de Machine Learning. Se comporta de 8 pasos principales:

1. Comprensión del contexto del problema y observación desde una perspectiva global.
2. Recopilación de datos.  
3. Exploración de los datos y obtención de introspectiva.  
4. Preparación de los datos para exponer de una manera más clara los patrones subyacentes de los datos a los algoritmos de Machine Learning.
5. Exploración de diversos modelos y acotado de los mejores.  
6. Ajuste fino de los modelos y combinación de los mismos para la obtención de una gran solución.  
7. Presentación de la solución.  
8. Lanzamiento, monitoreo, y mantenimiento del sistema.  

Obviamente, puedes adaptar esta checklist a a tus necesidades.

# Comprensión del contexto del problema y observación desde una perspectiva global  
1. Define el objetivo en términos de necesidades de negocio.  
2. ¿Cómo se utilizará tu solución?  
3. Cuáles son las soluciones o métodos actuales (si es que hay)?  
4. Plantea el marco del problema (aprendizaje supervisado/no supervisado, online/offline, etc.)  
5. ¿Cómo se debería medir el rendimiento?  
6. ¿Está la medida del rendimiento alineada con las necesidades de negocio?  
7. ¿Cuál sería el rendimiento mínimo necesitado para suplir las necesidades de negocio?  
8. ¿Existen problemas comparables? ¿Puedes reutilizar experiencia o recursos?  
9. ¿Existen expertos disponibles en dicho tema? 
10. ¿Cómo resolverías el problema manualmente?  
11. Lista las suposiciones que tú y otros habéis tenido hasta le momento.  
12. Verifica de ser posible dichas suposiciones.  

# Recopilación de datos.   
Nota: Automatiza este aspecto lo máximo posible para poder mantener los datos lo más actualizados posible.  

1. Lista los datos que necesitas y con qué volumen.  
2. Busca y documenta dónde se pueden obtener dichos datos.
3. Comprueba cuánto espacio necesitarías.  
4. Comprueba las obligaciones legales y obten la autorización si es necesario.  
5. Consigue acceso autorizado.  
6. Crea un entorno de trabajo (con suficiente capacidad de almacenamiento).  
7. Obten los datos.  
8. Convierte los datos a un formato fácilmente manipulable (sin cambiar los propios datos).  
9. Asegurate de que la información sensible esté eliminada o protegida (por ejemplo, anonimizada). 
10. Comprueba el tamaño y tipo de dato (series temporales, muestras, datos geográficos, etc.).  
11. Muestrea un conjunto de datos de prueba, ponlo aparte y no lo mires nunca (¡nada de spoilers!).    

# Exploración de los datos.
Nota: Intenta obtener la opinión de un experto en el campo para estos pasos.  

1. Crea una copia de los datos para su exploración (sampling it down to a manageable size if necessary).
2. Crea un notebook de Jupyter para registrar tu exploración.  
3. Estudia cada uno de los atributos y sus características:  
    - Nombre
    - Tipo (categórico, discreto/continuo, delimitado/ilimitado, texto, estructurado, etc.)
    - % de valores faltantes  
    - Si son ruidosos y tipo de ruido (estocástico, valores atípicos, errores de redondeo, etc.)
    - ¿Es posiblemente útil para resolver el problema?  
    - Tipo de distribución (Gaussiana, uniforme, logarítmica, etc.)
4. Para tareas de aprendizaje supervisado, identica el/los atributo/s objetivo.
5. Visualiza los datos.  
6. Estudia correlaciones entre atributos.  
7. Estudia cómo resolverías el problema manualmente.  
8. Identifica las transformaciones prometedoras que quizás querrías aplicar.  
9. Identifica datos adicionales que podrían resultar útiles (si existen trata de recopilarlos volviendo al paso de recopilación de datos).  
10. Documenta todo lo que has aprendido.  

# Preparación de los datos.
Notas:    
- Trabaja en copias de los datos (manten el dataset original intacto).
- Escribe funciones para todas las transformaciones de datos que apliques, por cinco razones:  
    - Para que puedas preparar fácilmente los datos cada vez que obtengas un dataset actualizado
    - Para poder aplicar estas transformaciones en futuros proyectos  
    - Para limpiar y preparar el conjunto de datos de prueba 
    - Para limpiar y preparar nuevas instancias de datos  
    - Para hacer más fácil el tratamiento de tus opciones de preparación como hiperparámetros 

1. Limpieza de datos:  
    - Soluciona o elimina los datos atípicos (opcional).  
    - Rellena los datos faltantes (por ejemplo, con cero, la media, la mediana..), o elimina sus filas (o columnas).  
2. Selección de atributos (opcional):  
    - Elimina los atributos que no provean información útil para la tarea.  
3. Ingeniería de atributos, donde sea apropiado:  
    - Discretizar atributos continuos.  
    - Decomponer atributos (por ejemplo, categóricas, fecha/hora, etc.).  
    - Crea transformaciones prometedoras de los atributos (por ejemplo, log(x), sqrt(x), x^2, etc.).
    - Agrega atributos en nuevas atributos prometedores.  
4. Escalado de atributos: Estandarización o normalización de atributos.  

# Acotado de los modelos más prometedores
Notas: 
- Si los datos son enormes, quizás querrías muestrear los datos en datasets de entrenamiento más pequeños para poder entrenar muchos modelos diferentes en un tiempo razonable (ten en cuenta que esto penaliza modelos complejos como grandes redes neuronales o Bosques Aleatorios).  
- Una vez más, intenta automatizar estos pasos lo máximo posible.    

1. Entrena muchos modelos rápidos y en sucio de diferentes categorías: lineal, naive, Bayes, SVM, Bosques Aleatorios, redes neuronales, etc.) usando parámetros estándar.
2. Mide y compara sus rendimientos.  
    - Para cada modelo, usa Validación cruzada K-Fold y computa la media y desviación típica de su rendimiento. 
3. Analiza las variables más significativas para cada algoritmo.  
4. Analiza los tipos de errores que cometen los modelos.  
    - ¿Qué datos utilizaría un humano para evitar esos errores?  
5. Ten una ronda rápida de selección e ingeniería de atributos.  
6. Repite una o dos veces más los pasos previos.
7. Acota una lista de entre los tres y cinco modelos más prometedores, preferiblemente modelos qué cometen errores de distinto tipo.  

# Ajuste fino del sistema
Notas:  
- Querrás usar tantos datos como sea posible para este paso, especialmente cuando nos vamos acercando al final del ajuste fino.   
- Como siempre automatiza lo que puedas.    

1. Ajusta los hiperparámetros usando validación cruzada.  
Trata tus opciones de transformación de datos como hiperparámetros, especialmente cuando no estés seguro de ellos (por ejemplo, ¿debería reemplezar los atributos faltantes con zero o con la mediana, o debería simplemente eliminar las filas?).
    - A no ser que haya pocos valores de hiperparámetros que explorar, elije búsqueda aleatoria sobre búsqueda en grid. Si el entrenamiento es demasiado largo, quizás prefieras un enfoque de If training is very long, you may prefer a optimización bayesiana (Por ejemplo, usando un proceso Gaussiano previo, como es descrito por Jasper Snoek, Hugo Larochelle, y Ryan Adams ([https://goo.gl/PEFfGr](https://goo.gl/PEFfGr)))  
2. Prueba métodos Conjuntos. Combinar tus mejores modelos a menudo da mejores resultados que utilizarlos individualmente.  
3. Una vez estés seguro de tu modelo final, mide su rendimiento con el conjunto de datos de prueba para estimar el error de generalización.

> No ajustes tu modelo después de medir el error de generalización: sólo conseguirías sobreajustar el conjunto de datos de prueba.  
  
# Presentación de la solución
1. Documenta lo que has hecho.  
2. Crea una buena presentación.  
    - Asegúrate de destacar la visión general del problema primero.  
3. Explica por qué tu solución satisface el objetivo de negocio.  
4. No te olvides de presentar puntos interesantes que notaste durante el proceso.  
    - Describe lo que funcionó y lo que no.  
    - Lista tus suposiciones y las limitaciones de tu sistema.  
5. Asegurate de que tus descubrimientos clave son comunicados con visualizaciones claras y afirmaciones fáciles de recordar (Por ejemplo, "La media de ingresos de la zona es el predictor número uno del valor de una vivienda").  

# Lanzamiento  
1. Prepara tu solución para producción (conéctate a entradas de datos de producción, escribe tests unitarios, etc.).  
2. Escribe código de monitoreo para comprobar el rendimiento en vivo de tu sistema en intervalos regulares y lanzar alertas en caso de caídas.  
    - Ten cuidado con la degradación lenta: los modelos tienden a "podrirse" conforme los datos evolucionan.   
    - Medir el rendimiento podría requerir de un pipeline con intervención humana (Por ejemplo, mediante un servicio de crowdsourcing o mediante una validación posterior de los resultados por el cliente).  
    - Monitorea también la calidad de tus entradas (Por ejemplo, un sensor dañado enviando valores aleatorios, la producción de otro miembro del equipo parándose). Esto es particularmente importante para sistemas de aprendiza en línea.  
3. Reentrena tu modelo regularmente con datos actualizados (automatiza esto lo máximo posible).  



