
<H1 aling="center"> 
National Technological Institute of Mexico Technological Of Tijuana
Academic Subdirectorate
Systems and Computing Department</H1>

<H3 aling="center">SEMESTER:</H3><H4 aling="center">January - June 2020</H4>
<H3 aling="center">CAREER:</H3>
<H4 aling="center">Ing. Information and Communication Technologies</H4>
<H3 aling="center">CLASS:</H3>
<H4 aling="center">Big Data</H4>
<H3 aling="center">UNIT TO EVALUATE:</H3>
<H4 aling="center">Unit 1</H4>
<H3 aling="center">Student's name</H3>
<H4 aling="center">
 Garcia Bautista Ana Laura # 15210793 

 Enciso Maldonado Aileen Yurely #15210329</H4> 

<H3 aling="center">Teacher (A):</H3>
<H4 aling="center">Romero Hernández José Christian </H4>

<H3> Units Section</H3>  

 <li type="type="square""><a href="https://github.com/anagarciabautista/DatosMasivos/tree/Unidad_2" target="_blank">Unit 2</a></li>
 <ul>
 
 <li type="circle"><a href="https://github.com/anagarciabautista/DatosMasivos/tree/Unidad_2/Evaluation" target="_blank">
Evaluation</a></li>
 </u>
 
 <H3>Table of Contents</H3>
 
[1.-DecisionTreeClassifier](#DecisionTreeClassifier)

[2.-RandaomForestClassifier](#RandaomForestClassifier)

[3.-GradientBoostedTreeClassifier](#GradientBoostedTreeClassifie)

[4.-MultilayerPerceptionClasifier](#MultilayerPerceptionClasifier)

[5.-LinearSupportVectorMachine](#LinearSupportVectorMachine)

[6.-OneVSRestClassifier](#OneVSRestClassifier)

[7.-NaiveBayes](#NaiveBayes)

[8.-Practice1](#Practice1)

[9.-Practice2](#Practice2)

[10.-Research1](#Research1)

[11.-Research2](#Research2)

[12.-Research3](#Research3)
 
# DecisionTreeClassifier
  
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center"> 
 
 1.- Se agregan las librerias del algortimo
 

 2.- Todas las clases y métodos que se utilizaran en el modelo
 
 3.- Transformers: Convierte un DataFrame  en otro agregando mas columnas, Estimators fit() que produce el modelo
 
 4.- Simétricamente a StringIndexer, IndexToString
 
 5.- Se instancia los DataFrame en la variable "data", el archivo debe estar estructurado al formato de trabajo
 
 6.- Se agrega una columna de indices, y se transformaran a datos numericos, para poder manipularlos
 
 7.- Se declararan 2 arreglos, se reparten de forma aleatoria
 
 8.- Se declara el Clasificador de árbol de decisión y se le agrega la columna que sera las etiquetas (indices) y los valores que cada   respectivo indice (caracteristicas)
 
 9.- Convierte las etiquetas indexadas a las originales
 
 10.- Crea el DT pipeline Agregando los index, label y el arbol juntos
 
 11.- Se entrena el modelo con los datos del arreglo "trainingData" que es el 70% de los datos totales
 
 12.-Se hacen las predicciones al tomar los datos sobrantes que se llevo "testData" que es el 30%
 
 13.- Se manda a imprimir la etiqueta, sus respectivos valores y la prediccion de la etiqueta
 
 14.- Evalua el modelo y retorna la  métrica escalar
 
 15.- La variable "accuracy" tomara la acertación que hubo respecto a "predictedLabel" y "label"
 
 16.- Se manda a imprimir el resultado de error con respecto a la exactitud
 
 17.- Se guarda en la variable
 
 18.- imprime el arbol de decisiones</H5>

    import org.apache.spark.ml.Pipeline 
    import org.apache.spark.ml.classification.DecisionTreeClassificationModel 
    import org.apache.spark.ml.classification.DecisionTreeClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer} 

    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)         features with > 4 distinct values are treated as continuous.

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    predictions.select("predictedLabel", "label", "features").show(5)

    val evaluator = new    MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)

    println(s"Test Error = ${(1.0 - accuracy)}")

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]

    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
    
 
 # RandaomForestClassifie
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center"> 
 
1.- Se agregan las librerias del algortimo


2.- Se carga y se convierte en un DataFrame.

3.- Se ajusta a todo el conjunto de datos para incluir todas las etiquetas en el índice.

4.- Se establecen el maxCategories para que las entidades con > 4 valores distintos se traten como continuas.

5.- Divide los datos en conjuntos de entrenamiento y prueba (30% para pruebas).

6.- Entrena un modelo RandomForest.

7.- Convierte las etiquetas indexadas de nuevo a etiquetas originales.

8.- Convierte las etiquetas indexadas de nuevo a etiquetas originales.

9.- Indicadores de cadena y bosque en una tubería.

10.- Indicadores de cadena y bosque en una tubería.

11.- Modelo de tren. Esto también ejecuta los indexadores.

12.- Hacer predicciones.

13.- Seleccione filas de ejemplo para mostrar.

14.- Seleccione (predicción, etiqueta verdadera) y calcule el error de prueba.</H5>

    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

    val data = spark.read.format("libsvm").load("./sample_libsvm_data.txt")data.show()

    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    predictions.select("predictedLabel", "label", "features").show(5)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")

    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel] 
    println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
 
 
 # GradientBoostedTreeClassifie
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center">
 
 1.- Se agregan las librerias del algortimo
 
 
 2.- Se carga los datos en la variable "data" en formato "libsvm"
 
 3.- Se agrega una nueva columna "IndexLabel" que tendra todos los datos de la columna "label"
 
 4.- Se agrega una nueva columna "indexedFeatures" que tendra todos los datos de la columna "features"
 
 5.- Se declaran dos arreglos; "trainingData" y "testData" de los cuales tendran 70% y 30% de los datos que fueron declarados en la  variable "data"
 
 6.- Se declara el modelo y se agregan como parametros "indexedLabel" y "indexedFeatures", que son las etiquetas de cada clase
y las caracteristicas de esa clase

 7.- Se convierten las "indexedLabel" a las etiquetas originales
 
 8.- Se declara el objeto "pipeline" en donde nos ayudara a pasar el codigo por estados, estos mismos estan declarados despues de "Array"

9.- Se entrena el modelo con los datos de entrenamiento

10.- Se hacen las predicciones con el modelos ya entrenado y con los datos de prueba que representan el 30%

11.- Se mandan a imprimir o se seleccionan algunas columnas y se muestran solo las primerias 5

12.- Se evalua la precision y se agrega a una variable "accuracy"

13.- Se manda a imprimir el error de precision del modelo

14.- Se manda a imprimir el arbol por medio de condicionales "if and else"</H5>

    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")

    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    predictions.select("predictedLabel", "label", "features").show(5)

    val evaluator = new    MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)

    println(s"Test Error = ${1.0 - accuracy}")

    val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
    println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
  
 # MultilayerPerceptionClasifier
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center"> 
 
1.- Se agregan las librerias del algortimo


2.- Se cargan los datos en la variable "data" en un formato "libsvm"

3.- Se declara un variable llamda "splits" donde se hacen los cortes de forma aleatoria de los datos de la variable "data"

4.- Se declara la variable "train" donde con ayuda de "splits" tendra el primer parametro que es el 60% de los datos cortados

5.- Se declara la variable "test" donde con ayuda de "splits" tendra el primer parametro que es el 40% de los datos cortados

6.- Se especifican las capas de la red neuronal

7.- Capa de entrada de tamaño 4 (características), dos intermedios de tamaño 5 y 4 y salida de tamaño 3 (clases)

8.- Se declara el modelo y se agregan los parametros necesarios para su funcionamiento

9.- Se entrena el modelo con los datos de entrenamiento

10.- Se evalua y despliega resultados

11.- Se imprime el error de la precision</H5> 

    import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    val data = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")

    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)

    val train = splits(0)

    val test = splits(1)

    val layers = Array[Int](4, 5, 4, 3)

    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

    val model = trainer.fit(train)

    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
 
 # LinearSupportVectorMachine
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center">
 
 1.- importamos la libreria BAM!!! 

 2.-Carga los datos de entrenamiento de la base de datos de prueba
 
 3.- Establede el maximo de interaciones y establecemos una region de parametros de 0.1
 
 4.- Ajusta el modelo  a los datos de entrenamiento
 
 5.- Imprime los coeficientes e interceta los valores para Linear SVM, maximixa la distancia entre los valores.
 
 6.-  Imprime un Super BAM!!!</H5>

     import org.apache.spark.ml.classification.LinearSVC

    val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val lsvc = new LinearSVC()
    .setMaxIter(10)
    .setRegParam(0.1)

    val lsvcModel = lsvc.fit(training)

    println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

    println("")

    println("SUPER BAM!!!")
 
 # OneVSRestClassifier
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center"> 
 
 1.- Se importan todas la librerias necesarias
 
 2.- Se agregan las librerias del algortimo
 
 3.- Se declararan 2 arreglos, uno tendra los datos de entrenamiento y el otro tendra
 
 4.- los datos de prueba, respectivamente fueron declarados como arreglos y tendran el 80 y 20 porciento de los datos totales
 
 5.- Se declara la variable "classifier" que hara la regresion
 
 6.- Se declara el modelo "OneVsRest"
 
 7.- Se entrena el modelo con los datos de entrenamiento
 
 8.- Se hacen las predicciones con los datos de prueba
 
 9.- Se declara el evaluador que tomara la precision del modelo y lo guardara en una variable metrica llamada "accuracy"
 
 10.- Se calcula el error del modelo con una simple resta</H5>

    import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    val inputData = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")

    val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

    val classifier = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)

    val ovr = new OneVsRest().setClassifier(classifier)

    val ovrModel = ovr.fit(train)

    val predictions = ovrModel.transform(test)

    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1 - accuracy}")
  
 # NaiveBayes
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center"> 

1.- Se agregan las librerias del algortimo

2.- Cargar los datos almacenados en formato LIBSVM como un DataFrame.

3.- Dividir los datos en conjuntos de entrenamiento y prueba (30% para pruebas)

4.- Entrena un modelo NaiveBayes.

5.- Seleccione filas de ejemplo para mostrar.

6.- Seleccionar (predicción, etiqueta verdadera) y calcular error de prueba</H5>

    import org.apache.spark.ml.classification.NaiveBayes
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

    val model = new NaiveBayes().fit(trainingData)

     val predictions = model.transform(testData)
     predictions.show()

     val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")
 
 # Practice1
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center">

1.- impoetamos la paqueteria de regresion lineal

2.- Cargar datos de entrenamiento

3.- Ajustar el modelo

4.- Imprime los coeficientes e intercepta para regresión lineal

5.- Resuma el modelo sobre el conjunto de entrenamiento e imprima algunas métricas</H5>

    import org.apache.spark.ml.regression.LinearRegression

    val training =spark.read.format("libsvm").load("data/mllib/sample_linear_regression_data.txt")
    val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

    val lrModel = lr.fit(training)

    println(s"Coefficients: ${lrModel.coefficients} Intercept:${lrModel.intercept}")

    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory:[${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
 
 
 # Practice2
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center">

1.- importamos Librerias A Utilizar

2.- Elimina varios avisos de warnings/errores inecesarios

3.- Iniciamos sesion en spark

4.- Creacion del dataframe para cargar el archivo csv

5.- Imprimimos el esquema del dataframe para visualizarlo

6.- Imprime la primera linea de datos del csv

7.- Tomamos nuestros datos mas relevantes a una variables y tomamos clicked on ad como nuestra label

8.- Generamos nuestro vector de ensamble en un arrengo donde tomamos nuestros features

9.- Utilizamos la regresion lineal en nuestros datos con un 70% y 30% de datos.

10.-Creacion del modelo

11.- Resultados de las pruebas con nuestro modelo

12.- Imprimimos nuestras metricas y la accuaricy de los calculos</H5>

    import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
    import org.apache.spark.mllib.evaluation.MulticlassMetrics
    import org.apache.spark.ml.classification.LogisticRegression 
    import org.apache.spark.sql.types.DateType
    import org.apache.spark.ml.linalg.Vectors
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.ml.Pipeline
    import org.apache.log4j._
 
    Logger.getLogger("org").setLevel(Level.ERROR)
 
    val spark = SparkSession.builder().getOrCreate()
 
    val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")
 
    data.printSchema()
 
    data.head(1)
    data.select("Clicked on Ad").show()
    val timedata = data.withColumn("Hour",hour(data("Timestamp")))
 
    val logregdataall = timedata.select(data("Clicked on Ad").as("label"),$"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily Internet Usage",$"Hour",$"Male")
    val feature_data = data.select($"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily Internet Usage",$"Timestamp",$"Male")
    val logregdataal = (data.withColumn("Hour",hour(data("Timestamp")))
    val logregdataal = logregdataall.na.drop()
 
    val assembler = new VectorAssembler().setInputCols(Array("Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Hour","Male")).setOutputCol("features")
 
    val Array(training, test) = logregdataall.randomSplit(Array(0.7, 0.3), seed = 12345)
    val lr = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(assembler,lr))
 
    val model = pipeline.fit(training)
 
    val results = model.transform(test)
    val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)
 
    println("Confusion matrix:")
    println(metrics.confusionMatrix)
    metrics.accuracy
  
 # Research1
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

<H4>Principales tipos de algoritmos de Machine Learning</H4>

<H5 aling="center"> 

Algoritmos de Regresión

Los algoritmos de Regresión modelan la relación entre distintas variables (features) utilizando una medida de error que se intentará minimizar en un proceso iterativo para poder realizar predicciones «lo más acertadas posible». Se utilizan mucho en el análisis estadístico. El ejemplo clásico es la predicción de precios de Inmuebles a partir de sus características: cantidad de ambientes del piso, barrio, distancia al centro, metros cuadrados del piso, etc.
Los Algoritmos más usados de Regresión son:
•	Regresión Lineal. 
•	Regresión Logística.


Algoritmos basados en Instancia

Son Modelos de Aprendizaje para problemas de decisión con instancias o ejemplos (muestras) de datos de entrenamiento que son importantes o requeridos por el modelo.
También son llamados Algoritmos «Ganador se lleva todo» y aprendizaje basado-en-memoria en el que se crea un modelo a partir de una base de datos y se agregan nuevos datos comparando su similitud con las muestras ya existentes para encontrar «la mejor pareja» y hacer la predicción.
Los Algoritmos basados en instancia más usados son:
•	k-Nearest Neighbor (kNN).
•	Self-Organizing Map.

Algoritmos de Árbol de Decisión

Modelan la toma de Decisión basado en los valores actuales (reales) de los atributos que tienen nuestros datos. Se utilizan sobre todo para clasificación de información, bifurcando y modelando los posibles caminos tomados y su probabilidad de ocurrencia para mejorar su precisión. Una vez armados, los arboles de decisión ejecutan muy rápido para obtener resultados.
Los Algoritmos de árbol de decisión más usados son:
•	Arboles de Clasificación y Regresión (CART) 
•	Decisión de Arbol condicional
•	Random Forest 

Algoritmos Bayesianos

Son algoritmos que utilizan explícitamente el Teorema de Bayes de probabilidad para problemas de Clasificación y Regresión.
Los más utilizados son:
•	Naive Bayes
•	Gaussian Naive Bayes 
•	Multinomial Naive Bayes
•	Bayesian Network

Algoritmos de Clustering (agrupación)

Se utilizan para agrupar datos existentes de los que desconocemos sus características en común o queremos descubrirlas.
Estos métodos intentan crear «puntos centrales» y jerarquías para diferenciar grupos y descubrir características comunes por cercanía.
Los más utilizados son:
•	K-Means 
•	K-Medians
•	Hierarchical Clustering

Algoritmos de Redes Neuronales

Son algoritmos y estructuras inspirados en las funciones biológicas de las redes neuronales. Se suelen utilizar para problemas de 
Clasificación y Regresión, pero realmente tienen un gran potencial para resolver multitud de problemáticas. Son muy buenas para detectar patrones. Las Redes Neuronales Artificiales requieren mucha capacidad de procesamiento y memoria y estuvieron muy limitadas por la tecnología del pasado hasta estos últimos años en los que resurgieron con mucha fuerza dando lugar al Aprendizaje Profundo (se detalla más adelante).

Las redes neuronales básicas y clásicas son:	
•	Compuerta XOR.
•	Perceptron.
•	Back-Propagation.
•	Hopfield Network 
•	MLP: Multi Layered Perceptron.

Algoritmos de Aprendizaje Profundo

Son la evolución de las Redes Neuronales Artificiales que aprovechan el abaratamiento de la tecnología y la mayor capacidad de ejecución, memoria y disco para explotar gran cantidad de datos en enormes redes neuronales interconectarlas en diversas capas que pueden ejecutar en paralelo para realizar cálculos. Comprende mejor Deep Learning con esta guía rápida que escribí.
Los algoritmos más populares de Deep Learning son:
•	Convolutional Neural Networks.
•	Long Short Term Memory Neural Networks.

Algoritmos de Reducción de Dimensión

Buscan explotar la estructura existente de manera no supervisada para simplificar los datos y reducirlos o comprimirlos.
Son útiles para visualizar datos o para simplificar el conjunto de variables que luego pueda usar un algoritmo supervisado.
Los más utilizados son:
•	Principal Component Analysis (PCA).
•	t-SNE (próximamente artículo). 

Procesamiento del Lenguaje Natural (NLP)

El Natural Language Processing es una mezcla entre DataScience, Machine Learning y Lingüística. Tiene como objetivo comprender el lenguaje humano. Tanto en textos como en discurso/voz. Desde analizar sintáctica ó gramáticamente miles contenidos, clasificar automáticamente en temas, los chatbots y hasta generar poesía imitando a Shakespeare. También es común utilizarlo para el Análisis de Sentimientos en redes sociales, (por ejemplo, con respecto a un político) y la traducción automática entre idiomas. Asistentes como Siri, Cortana y la posibilidad de preguntar y obtener respuest </H5>
  
 # Research2
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

<H4>VectorAssembler, Vector, RMSE: Root Mean Square Error</H4>

<H5 aling="center"> 

VectorAssembler

VectorAssembleres un transformador que combina una lista dada de columnas en una sola columna vectorial. Es útil para combinar características sin procesar y características generadas por diferentes transformadores de características en un solo vector de características, con el fin de entrenar modelos ML como la regresión logística y los árboles de decisión. VectorAssembleracepta los siguientes tipos de columnas de entrada: todos los tipos numéricos, tipo booleano y tipo vectorial. En cada fila, los valores de las columnas de entrada se concatenarán en un vector en el orden especificado.

Vector

En física, un vector1 es un ente matemático como la recta o el plano. Un vector se representa mediante un segmento de recta, orientado dentro del espacio euclidiano tridimensional. El vector tiene 3 elementos: módulo, dirección y sentido.  Los vectores nos permiten representar magnitudes físicas vectoriales, como las mencionadas líneas abajo.
En matemáticas se define vector como un elemento de un espacio vectorial. Esta noción es más abstracta y para muchos espacios vectoriales no es posible representar sus vectores mediante el módulo y la dirección. En particular los espacios de dimensión infinita sin producto escalar no son representables de ese modo. Los vectores en un espacio euclídeo se pueden representar geométricamente como segmentos de recta R, en el plano R2 o en el espacio R3
Algunos ejemplos de magnitudes físicas que son magnitudes vectoriales: la velocidad con que se desplaza un móvil, ya que no queda definida tan solo por su módulo que es lo que marca el velocímetro, en el caso de un automóvil, sino que se requiere indicar la dirección (hacia donde se dirige), la fuerza que actúa sobre un objeto, ya que su efecto depende además de su magnitud o módulo, de la dirección en la que actúa; también, el desplazamiento de un objeto, pues es necesario definir el punto inicial y final del movimiento.

RMSE: error cuadrático medio de raíz

El error cuadrático medio (RMSE) es la desviación estándar de los residuos (errores de predicción). Los residuos son una medida de qué tan lejos están los puntos de datos de la línea de regresión; RMSE es una medida de la dispersión de estos residuos. En otras palabras, le dice qué tan concentrados están los datos alrededor de la línea de mejor ajuste. El error cuadrático medio se usa comúnmente en climatología, predicción y análisis de regresión para verificar resultados experimentales.
La fórmula es:

Donde:
•	f = pronósticos (valores esperados o resultados desconocidos),
•	o= valores observados (resultados conocidos).

La barra sobre las diferencias al cuadrado es la media (similar a x̄). La misma fórmula se puede escribir con la siguiente notación, ligeramente diferente (Barnston, 1992):

Donde:
Σ = sumatoria ("suma")
(z f i - Z o i ) Sup > 2 = diferencias, al cuadrado
N = tamaño de muestra.

Puede usar la fórmula con la que se sienta más cómodo, ya que ambos hacen lo mismo. Si no le gustan las fórmulas, puede encontrar el RMSE:
•	Cuadrando los residuos.
•	Encontrar el promedio de los residuos.
•	Tomando la raíz cuadrada del resultado.

Dicho esto, esto puede ser mucho cálculo, dependiendo de qué tan grandes sean sus datos. Un atajo para encontrar el error cuadrático medio es:

Donde SD y es la desviación estándar de Y.
Cuando se utilizan observaciones y pronósticos estandarizados como entradas RMSE, existe una relación directa con el coeficiente de correlación. Por ejemplo, si el coeficiente de correlación es 1, el RMSE será 0, porque todos los puntos se encuentran en la línea de regresión (y, por lo tanto, no hay errores)
  
</H5>  

# Research3
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>
 
 <H4>Pipeline y Matriz De Confusión</H4>

 <H5 aling="center"> 

Definición De Pipeline

Pipeline es un término inglés que puede traducirse como “tubería”. Aunque no forma parte del diccionario de la Real Academia Española 
(RAE), el concepto se utiliza en nuestra lengua para hacer referencia a una arquitectura de la informática.

En Arquitectura de Computadoras

El pipeline es una técnica para implementar simultaneidad a nivel de instrucciones dentro de un solo procesador. Pipelining intenta mantener ocupada a cada parte del procesador, dividiendo las instrucciones entrantes en una serie de pasos secuenciales, que se realizan por diferentes unidades del procesador que trabajan de forma simultánea. Aumenta el rendimiento de la CPU a una velocidad de reloj determinada, aunque puede aumentar la latencia debido a la sobrecarga adicional del proceso de pipeline en sí.

En informática y programación

La arquitectura en pipeline (basada en filtros) consiste en ir transformando un flujo de datos en un proceso comprendido por varias fases secuenciales, siendo la entrada de cada una la salida de la anterior. Esta arquitectura es muy común en el desarrollo de programas para el intérprete de comandos, ya que se pueden conectar comandos fácilmente con tuberías (pipe).
También es una arquitectura muy natural en el paradigma de programación funcional, ya que equivale a la composición de funciones matemáticas.

Matriz de confusión

En el campo de la inteligencia artificial una matriz de confusión es una herramienta que permite la visualización del desempeño de un algoritmo que se emplea en aprendizaje supervisado. Cada columna de la matriz representa el número de predicciones de cada clase, mientras que cada fila representa a las instancias en la clase real. Uno de los beneficios de las matrices de confusión es que facilitan ver si el sistema está confundiendo dos clases.
Si en los datos de entrada el número de muestras de clases diferentes cambia mucho la tasa de error del clasificador no es representativa de lo bien que realiza la tarea el clasificador. Si por ejemplo hay 990 muestras de la clase 1 y sólo 10 de la clase 2, el clasificador puede tener fácilmente un sesgo hacia la clase  1. Si el clasificador clasifica todas las muestras como clase 1 su precisión será del 99%. Esto no significa que sea un buen    clasificador, pues tuvo un 100% de error en la clasificación de las muestras de la clase 2.

En el matriz ejemplo que aparece a continuación, de 8 gatos reales, el sistema predijo que tres eran perros y de seis perros predijo que uno era un conejo y dos eran gatos. A partir de la matriz se puede ver que el sistema tiene problemas distinguiendo entre gatos y perros, pero que puede distinguir razonablemente bien entre conejos y otros animales. 
 </H5> 
