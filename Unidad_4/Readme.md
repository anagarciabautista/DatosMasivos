 <H1> Test </H1>
 
 <div align="center"><H2>Index</H2>
 <H3>Introduction</H3>
 <H3>theoretical framework</H3>
 <H3>Implementation</H3>
 <H3>Result</H3>
 <H3>Conclusion</H3>
 <H3>References</H3></div>
 
 <H1>INFORMATION</H1>
 
 <H2>INTRODUCTION</H2>

</H4><p align="justify">In this work, a data analysis is carried out, which is a massive data implementation most used today due to the amount of data that is handled today, which is a large amount of data to analyze, so when comparing the Logistic Regression algorithms, Decision three, Multilayer Perceptron, are the algorithms that we intend to use to carry out this investigation. Based on these algorithms, it is intended to make an analysis of the data of a banking system to analyze which algorithm can be more efficient for the banking system based on its behavior. To carry out the comparison of the work, I use the Spark-Scala tool that helps us to have a better control of our data to use.</p></H4>

<H2>THEORETICAL FRAREWORK</H2>

<H4><p align="justify">Nowadays, data analysis is one of the most common when it is necessary to analyze large data, which is why in this data analysis of a banking system a supervised system classification will be made, which is defined as classes or base label to the data to analyze. In order to determine which algorithm favors the banking system, the time and pressure of each algorithm will be analyzed. Based on our work, the data provided by the banking system is a set of data that the banking system will analyze in order to have better control of its system.</p></H4>

<H3>SVM algorithm</H3>

<H4><p align="justify">SVMs are one of the most powerful machine learning techniques. It consists of building a hyperplane in a space of very high (or even infinite) dimensionality that separates the classes we have. A good separation between the classes will allow a correct classification of the new sample, that is, we need to find the maximum separation to the points closest to this hyperplane.

The objective of classification problems applying this type of supervised learning algorithms is as follows; Given a training set with its class labels, train an SVM to build a model that predicts the class of a new sample or test set.</p></H4>
<div align="center"><img src="https://1.bp.blogspot.com/_jqWZ1wzdpTQ/SNMNvTDeT9I/AAAAAAAAAAQ/YjJ5OKnD6bs/s320/Dibujo.JPG"></div>

<H3>Decision Three Algorithm</H3>

<H4><p align="justify">Decision Tree or Decision Tree Classification is a type of supervised learning algorithm that is mainly used in classification problems, although it works for categorical input and output variables as continuous.

Tree-based learning algorithms are considered one of the best and most widely used supervised learning methods. Tree-based methods power predictive models with high precision, stability, and ease of interpretation. Unlike linear models, they map nonlinear relationships quite well.</p></H4>
<div align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png"></div>

<H3>Logistic Regression algorithm<H/3>

<H4><p align="justify">Logistic Regression or Logistic Regression is a classification algorithm that is used to predict the probability of a categorical dependent variable. Like all regression analyzes, logistic regression is predictive analysis. It is used to describe data and explain the relationship between a dependent binary variable and one or more independent nominal, interval, interval, or ratio level variables.

This binary logistic model is used to estimate the probability of a binary response based on one or more independent or predictive variables. Lets say that the presence of a risk factor increases the probability of a result given a specific percentage.</p></H4>
<div align="center"><img src="https://i2.wp.com/ligdigonzalez.com/wp-content/uploads/2018/03/Logistic-Regression-1-300x119.png?resize=300%2C119"></div>

<H3>Perceptron Multilayer Algorithm</H3>

<H4><p align="justify">El perceptrón multicapa es el hola mundo del aprendizaje profundo: un buen lugar para comenzar cuando estás aprendiendo sobre el aprendizaje profundo.

Un perceptrón multicapa (MLP) es una red neuronal profunda y artificial. Está compuesto por más de un perceptrón. Están compuestos por una capa de entrada para recibir la señal, una capa de salida que toma una decisión o predicción sobre la entrada, y entre esos dos, un número arbitrario de capas ocultas que son el verdadero motor computacional del MLP. Los MLP con una capa oculta son capaces de aproximar cualquier función continua.</p></H4>
<div align="center"><img src="https://www.researchgate.net/publication/259319882/figure/fig3/AS:667043535810572@1536046980868/Structure-of-a-typical-3-layer-feed-forward-multilayer-perceptron-artificial-neural.png"></div>

<H2>IMPLEMENTATION</H2>
 
<H3>Apche Spark</H3>

<H4><p align="justify">Spark is an ultra-fast engine for storing, processing, and analyzing large volumes of data. It is open source and is managed by the Apache Software Foundation. Therefore the tool is known as Apache Spark and it is one of their most active projects</p></H4>
<div align="center"><img src="https://www.vermilion.com.co/img/clients/client-10.png"></div>

<H3>Visual Studio Code</H3>

<H4><p align="justify">This software was used as interpretation of scala language</p></H4>
<div align="center"><img src="https://sobrebits.com/wp-content/uploads/2018/10/Visual-Studio-Code-para-PowerShell.png"></div>

<H2>RESULT</H2>

<H3>SVM</H3>

<H4><p align="justify">The SVM code allows us to find a category associated with the data sets since this algorithm is a model based on data predictions and to be able to organize by categories.</p></H4>

    import org.apache.spark.ml.classification.LinearSVC
    val change = df.withColumnRenamed("y", "label")
    val ft = change.select("label","features")
    ft.show()
    val cs1 = ft.withColumn("1abel",when(col("label").equalTo("1"),0).otherwise(col("label")))
    val cs2 = cs1.withColumn("label",when(col("label").equalTo("2"),1).otherwise(col("label")))
    val cs3 = cs2.withColumn("label",'label.cast("Int"))
    val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
    val lsvcModel = lsvc.fit(cs3)
    println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

<H3>Decision Three</H3>

<H4><p align="justify">In this algorithm, it predicts us in a data set using decision trees based on these classifications, it predicts and categorizes the data depending on its classification by the algorithm.</p></H4>

    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.evaluation.RegressionEvaluator
    import org.apache.spark.ml.regression.DecisionTreeRegressionModel
    import org.apache.spark.ml.regression.DecisionTreeRegressor

    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(c55)
    val Array(trainingData, testData) = c55.randomSplit(Array(0.7, 0.3))
    val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
    val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))
    val model43 = pipeline.fit(trainingData)
    val predictions = model.transform(testData)
    predictions.select("prediction", "label", "features").show(5)

<H3>Logistic Regression</H3>

<H4><p align="justify">In this algorithm we will be able to see the statistical analysis of the linear regression that, like the others, will categorize the data depending on it, and in turn will limit us to a certain number of numbers, but it shows us how essential it is to be able to categorize and predict our data in linear sets.</p></H4>

    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
    val  e4= predictions.withColumn("prediction",'prediction.cast("Double"))
    val rmse = evaluator.evaluate(e4)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

<H3>Multilayer Perceptron<H/3>

<H4><p align="justify">In this algorithm, precision is carried out through layers, separating the data and categorizing them depending on the problem established by the data, since they cannot be separated in a line manner, but as mentioned that the categorization by layers and prediction of the data is carried out. easier and optimally.</p></H4>

    import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    val splits = categories.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)
    val layers = Array[Int](2, 1, 3, 3)
    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
    val  c4= categories.withColumn("prediction",'prediction.cast("Double"))
    val  c5= c4.withColumn("label",'prediction.cast("Double"))
    val predictionAndLabels = c5 .select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

<H2>CONCLUSION</H2>

<H4><p align="justify">As a result of our project based on machine learning algorithms, we witnessed that the most efficient algorithm is the Multiplayer Perception because it was the one that gave us the least time to analyze data and with the best perception when processing the dataset and classified us as 85% of the data used in a correct way this work was done for the massive data subject taught by the teacher José Christian Romero Hernández</p></H4>

 <H2>REFERENCES</H2>
 
<H4>
<p align="justify">I. SVM algorithms for Big Data problems, Yvonne Gala García, José Ramon Dorronsoro Ibero, September 25, 2013
https://repositorio.uam.es/bitstream/handle/10486/14108/66152_Yvonne_Gala_Garcia.pdf?sequence=1&isAllowed=y [1].

II. Supervised Learning: Logistic Regression, Ligdi González, March 22, 2018.
https://ligdigonzalez.com/aprendizaje-supervisado-logistic-regression/#:~:text=La%20regresi%C3%B3n%20log%C3%ADstica%20o%20Logistic,de%20una%20variable%20dependiente%20categ%C3%B3rica.&text=Permite%20decir%20que%20la%20presencia,resultado%20dado%20un%20porcentaje%20espec%C3%ADfico.[2]

III. Supervised Learning: Decision Tree Classification, Ligdi González, March 23, 2018. 
https://ligdigonzalez.com/aprendizaje-supervisado-decision-tree classification/#:~:text=%C3%81rbol%20de%20decisi%C3%B3n%20o%20Decisi%C3%B3n,y%20salida%20categ%C3%B3ricas%20como%20continuas.&text=Las%20ventajas%20que%20tiene%20este,F%C3%A1cil%20de%20entender[3].

IV. A Beginner's Guide to Multilayer Perceptrons (MLP), Chris Nicholso.https://pathmind.com/wiki/multilayer-perceptron [4].

V. What is Spark and how does it revolutionize Big Data and Machine Learning ?, Yhorman Sierra, 2018.
https://blog.mdcloud.es/que-es-spark-big-data-y-machine-learning/ </p></H4>

