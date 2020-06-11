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
<H4 aling="center">Unit 2</H4>
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
 
<H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>
 
 <H2> Test </H2>
 
 <H5 aling="center">
  
1.- Agregamos las lbrerias necesarias para trabajar con el algortimo Multilayer Perceptron.

2.- Del data set Iris.cvs, elaborar la limpieza de datos necesaria por medio de un scrip en scala spark, impportamos las librerias necesarias para la limpieza.

3.- Se cargan los datos del dataser iris.csv en la variable "data"

4.- Vemos el esquema para comprobar que todos los valores estan calsificados correctamente en el datset

5.- Se eliminan los campos null 

6.- Se declara un vector que se transforma los datos a la variable "features"

7.- Se transforman los features usando el dataframe

8.- Se declara un "StringIndexer" que transformada los datos en "species" en datos numericos 

9.- Ajustamos las especies indexadas con el vector features

10.- Con la variable "splits" hacemos un corte de forma aleatoria

11.- Se declara la variable "train" la cual tendra el 60% de los datos

12.- Se declara la variable "test" la cual tendra el 40% de los datos

13.- Se establece la configuracion de las capas para el modelo de redes neuronales artificiales

14.- Se configura el entrenador del algoritmo Multilayer con sus respectivos parametros

15.- Se entrena el modelo con los datos de entrenamiento

16.- Se prueban ya entrenado el modelo

17.- Se selecciona la prediccion y la etiqueta que seran guardado en la variable 

18.- Se muestran algunos datos 

19.- Se ejecuta la estimacion de la precision del modelo

20.-Se imprime el error del modelo</H5>

    import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
    import org.apache.spark.ml.linalg.Vectors

    val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")

    data.printSchema()

    val dataClean = data.na.drop()

     val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))

     val features = vectorFeatures.transform(dataClean)

     val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")

     val dataIndexed = speciesIndexer.fit(features).transform(features)

     val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)

      val train = splits(0)
 
     val test = splits(1)

    val layers = Array[Int](4, 2, 2, 3)

    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

    val model = trainer.fit(train)

    val result = model.transform(test)

    val predictionAndLabels = result.select("prediction", "label")

    predictionAndLabels.show()

     val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictionAndLabels)

    println(s"Test Error = ${(1.0 - accuracy)}")
