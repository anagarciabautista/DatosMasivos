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
 
<H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>
 
 <H2> Test </H2>
 
 <H5 aling="center">
  

1.- We add the necessary libraries to work with the Multilayer Perceptron algorithm.

2.- From the data set Iris.cvs, elaborate the necessary data cleaning by means of a scrip in scala spark, we import the necessary libraries for the cleaning.

3.- The data from the dataser iris.csv is loaded in the variable "data"

4.- We see the scheme to check that all the values ​​are correctly classified in the datset

5.- null fields are removed

6.- A vector is declared that transforms the data to the variable "features"

7.- The features are transformed using the dataframe

8.- A "StringIndexer" is declared that transforms the data in "species" into numerical data

9.- We adjust the indexed species with the vector features

10.- With the variable "splits" we make a cut randomly

11.- The variable "train" is declared which will have 60% of the data

12.- The variable "test" is declared which will have 40% of the data

13.- The configuration of the layers for the artificial neural network model is established.

14.- The Multilayer algorithm trainer is configured with their respective parameters

15.- The model is trained with the training data

16.- The model is tested already trained

17.- Select the prediction and the label that will be stored in the variable

18.- Some data is shown

19.- The estimation of the model precision is executed

20.-The model error is printed</H5>

    import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
    import org.apache.spark.ml.linalg.Vectors

    val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")

    data.printSchema()

    val dataClean = data.na.drop()

     val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width",
     "petal_length","petal_width")).setOutputCol("features"))

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

     val evaluator = new
     MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictionAndLabels)

    println(s"Test Error = ${(1.0 - accuracy)}")
