
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
 
 1.- The algorithm libraries are added

 2.- All the classes and methods that will be used in the model
 
 3.- Transformers: Convert one DataFrame to another by adding more columns, Estimators fit () that produces the model
 
 4.- Symmetrically to StringIndexer, IndexToString
 
 5.- The DataFrame is instantiated in the variable "data", the file must be structured in the work format
 
 6.- An index column is added, and they will be transformed to numerical data, to be able to manipulate them
 
 7.- 2 arrangements will be declared, they are distributed randomly
 
 8.- The Decision Tree Classifier is declared and the column that will be the labels (indexes) and the values that each respective index (characteristics) is added to it.
 
 9.- Convert the indexed labels to the original
 
 10.- Create the DT pipeline Adding the index, label and the tree together
 
 11.- The model is trained with the data from the "trainingData" arrangement, which is 70% of the total data
 
 12.-Predictions are made by taking the surplus data that was taken "testData" which is 30%
 
 13.- The label, its respective values ​​and the prediction of the label are sent to print
 
 14.- Evaluate the model and return the scalar metric
 
 15.- The variable "accuracy" will take the correctness that there was regarding "predictedLabel" and "label"
 
 16.- The error result regarding the accuracy is sent to print
 
 17.- It is saved in the variable
 
 18.- print the decision tree</H5>

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

    val evaluator = new
    MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)

    println(s"Test Error = ${(1.0 - accuracy)}")

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]

    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
    
 
 # RandaomForestClassifie
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center"> 
1.- The algorithm libraries are added

2.- It is loaded and becomes a DataFrame.

3.- It adjusts to the entire data set to include all the labels in the index.

4.- The maxCategories are established so that entities with> 4 different values ​​are treated as continuous.

5.- Divide the data into training and test sets (30% for tests).

6.- Train a RandomForest model.

7.- Convert the indexed labels back to original labels.

8.- Convert the indexed labels back to original labels.

9.- Chain and forest indicators in a pipeline.

10.- Chain and forest indicators in a pipeline.

11.- Train model. This also runs the indexers.

12.- Make predictions.

13.- Select example rows to display.

14.- Select (prediction, true label) and calculate the test error.</H5>

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

    val evaluator = new
    MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")

    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel] 
    println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
 
 
 # GradientBoostedTreeClassifie
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center">

1.- The algorithm libraries are added
 
 2.- The data is loaded in the variable "data" in "libsvm" format
 
 3.- A new column "IndexLabel" is added that will have all the data of the column "label"
 
 4.- A new "indexedFeatures" column is added that will have all the data in the "features" column
 
 5.- Two arrangements are declared; "trainingData" and "testData" of which will have 70% and 30% of the data that were declared in the   variable "data"
 
 6.- The model is declared and "indexedLabel" and "indexedFeatures" are added as parameters, which are the labels of each class
  and the characteristics of that class

 7.- The "indexedLabel" are converted to the original tags
 
 8.- The "pipeline" object is declared where it will help us to pass the code by states, these are declared after "Array"

9.- The model is trained with the training data

10.- Predictions are made with the model already trained and with the test data representing 30%

11.- Some columns are sent to print or selected and only the first 5 are shown

12.- Accuracy is evaluated and added to a variable "accuracy"

13.- The model precision error is sent to print

14.- The tree is sent to print by means of conditionals "if and else"</H5>

    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val gbt = new
    GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")

    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    predictions.select("predictedLabel", "label", "features").show(5)

    val evaluator = new
    MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)

    println(s"Test Error = ${1.0 - accuracy}")

    val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
    println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
  
 # MultilayerPerceptionClasifier
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center"> 
 
1.- The algorithm libraries are added

2.- The data is loaded in the variable "data" in a "libsvm" format

3.- A variable called "splits" is declared where the cuts of the data of the variable "data" are made at random.

4.- The variable "train" is declared where with the help of "splits" it will have the first parameter which is 60% of the cut data

5.- The variable "test" is declared where with the help of "splits" it will have the first parameter which is 40% of the cut data

6.- The layers of the neural network are specified

7.- Input layer of size 4 (characteristics), two intermediates of size 5 and 4 and output of size 3 (classes)

8.- The model is declared and the necessary parameters for its operation are added

9.- The model is trained with the training data

10.- Results are evaluated and displayed

11.- The precision error is printed</H5> 

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
 
1.- we import the BAM library !!!

 2.-Load the training data from the test database
 
 3.- Set the maximum number of interactions and establish a region of parameters of 0.1
 
 4.- Fit the model to the training data
 
 5.- Print the coefficients and intercept the values ​​for Linear SVM, maximizing the distance between the values.
 
 6.- Print a Super BAM !!!</H5>

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
 1.- All necessary libraries are imported
 
 2.- The algorithm libraries are added
 
 3.- 2 arrangements will be declared, one will have the training data and the other will have
 
 4.- the test data, respectively, were declared as fixes and will have 80 and 20 percent of the total data
 
 5.- The variable "classifier" that will make the regression is declared.
 
 6.- The "OneVsRest" model is declared
 
 7.- The model is trained with the training data
 
 8.- Predictions are made with test data
 
 9.- The evaluator is declared who will take the precision of the model and save it in a metric variable called "accuracy"
 
 10.- The model error is calculated with a simple subtraction</H5>

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
 
1.- The algorithm libraries are added

2.- Load the data stored in LIBSVM format as a DataFrame.

3.- Divide the data into training and test sets (30% for tests)

4.- Train a NaiveBayes model.

5.- Select example rows to display.

6.- Select (prediction, true label) and calculate test error</H5>

    import org.apache.spark.ml.classification.NaiveBayes
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

    val model = new NaiveBayes().fit(trainingData)

     val predictions = model.transform(testData)
     predictions.show()

     val evaluator = new
     MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")
 
 # Practice1
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center">

1.- we import the linear regression package

2.- Load training data

3.- Fit the model

4.- Print the coefficients and intercept for linear regression

5.- Summarize the model on the training set and print some metrics</H5>

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

1.- we import Libraries To Use

2.- Eliminate several warnings / unnecessary errors warnings

3.- We start session in spark

4.- Creation of the dataframe to load the csv file

5.- We print the schematic of the dataframe to visualize it

6.- Print the first line of csv data

7.- We take our most relevant data to a variable and we take clicked on ad as our label

8.- We generate our assembly vector in a rental where we take our features

9.- We use linear regression in our data with 70% and 30% data.

10.-Creation of the model

11.- Test results with our model

12.- We print our metrics and the accuaricy of the calculations</H5>

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
 
    val logregdataall = timedata.select(data("Clicked on Ad").as("label"),$"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily
    Internet Usage",$"Hour",$"Male")
    val feature_data = data.select($"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily Internet Usage",$"Timestamp",$"Male")
    val logregdataal = (data.withColumn("Hour",hour(data("Timestamp")))
    val logregdataal = logregdataall.na.drop()
 
    val assembler = new VectorAssembler().setInputCols(Array("Daily Time Spent on Site","Age","Area Income","Daily Internet
    Usage","Hour","Male")).setOutputCol("features")
 
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

<H4>Main types of Machine Learning algorithms</H4>

<H5 aling="center"> 

Regression algorithms

Regression algorithms model the relationship between different variables (features) using an error measure that will be tried to minimize in an iterative process in order to make predictions "as accurate as possible". They are widely used in statistical analysis. The classic example is the prediction of Real Estate prices based on their characteristics: number of rooms in the apartment, neighborhood, distance to the center, square meters of the apartment, etc.
The most used Regression Algorithms are:
•	Linear regression.
•	Logistic regression.


Instance-based algorithms

They are Learning Models for decision problems with instances or examples (samples) of training data that are important or required by the model.
They are also called "Winner Takes All" algorithms and memory-based learning in which a model is created from a database and new data is added comparing their similarity with existing samples to find "the best match »And make the prediction.
The most used instance-based algorithms are:
• k-Nearest Neighbor (kNN).
• Self-Organizing Map.

Decision Tree Algorithms

They model Decision making based on the current (actual) values ​​of the attributes that our data has. They are mainly used for information classification, bifurcating and modeling the possible paths taken and their probability of occurrence to improve their precision. Once assembled, decision trees run very fast to get results.
The most used decision tree algorithms are:
• Classification and Regression Trees (CART)
• Conditional Tree Decision
• Random Forest

Bayesian algorithms

They are algorithms that explicitly use the Bayes Probability Theorem for Classification and Regression problems.
The most used are:
• Naive Bayes
• Gaussian Naive Bayes
• Multinomial Naive Bayes
• Bayesian Network

Clustering algorithms (grouping)

They are used to group existing data of which we do not know their characteristics in common or want to discover them.
These methods attempt to create "center points" and hierarchies to differentiate groups and discover common characteristics by closeness.
The most used are:
• K-Means
• K-Medians
• Hierarchical Clustering

Neural Network Algorithms

They are algorithms and structures inspired by the biological functions of neural networks. They are usually used for problems of
Classification and Regression, but they really have great potential to solve a multitude of problems. They are very good at spotting patterns. Artificial Neural Networks require a lot of memory and processing capacity and were very limited by the technology of the past until these last years in which they re-emerged with great force giving rise to Deep Learning (detailed below).

The basic and classic neural networks are:
• XOR gate.
• Perceptron.
• Back-Propagation.
• Hopfield Network
• MLP: Multi Layered Perceptron.

Deep Learning Algorithms

They are the evolution of Artificial Neural Networks that take advantage of the cheaper technology and the greater execution capacity, memory and disk to exploit large amounts of data in huge neural networks, interconnecting them in different layers that can be executed in parallel to perform calculations. Get a better understanding of Deep Learning with this quick guide I wrote.
The most popular Deep Learning algorithms are:
• Convolutional Neural Networks.
• Long Short Term Memory Neural Networks.

Dimension Reduction Algorithms

They seek to exploit the existing structure in an unsupervised way to simplify the data and reduce or compress it.
They are useful for visualizing data or for simplifying the set of variables that can then be used by a supervised algorithm.
The most used are:
• Principal Component Analysis (PCA).
• t-SNE (soon article).

Natural Language Processing (NLP)

Natural Language Processing is a mix between DataScience, Machine Learning and Linguistics. It aims to understand human language. Both in texts and in speech / voice. From analyzing syntactically or grammatically thousands of contents, automatically classifying into themes, chatbots and even generating poetry imitating Shakespeare. It is also common to use it for Sentiment Analysis on social media, (for example, regarding a politician) and machine translation between languages. Assistants like Siri, Cortana and the possibility to ask and get answers </H5>
  
 # Research2
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

<H4>VectorAssembler, Vector, RMSE: Root Mean Square Error</H4>

<H5 aling="center"> 

VectorAssembler

VectorAssembleres a transformer that combines a given list of columns into a single vector column. It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models such as logistic regression and decision trees. VectorAssemble accepts the following types of input columns: all numeric types, boolean type and vector type. In each row, the values ​​in the input columns will be concatenated into a vector in the specified order.

Vector

In physics, a vector1 is a mathematical entity like the line or the plane. A vector is represented by a line segment, oriented within the three-dimensional Euclidean space. The vector has 3 elements: module, direction and sense. Vectors allow us to represent vector physical quantities, such as those mentioned below.
In mathematics vector is defined as an element of a vector space. This notion is more abstract and for many vector spaces it is not possible to represent their vectors by modulus and direction. In particular, spaces of infinite dimension without scalar product are not representable in this way. Vectors in a Euclidean space can be represented geometrically as line segments R, in plane R2 or in space R3
Some examples of physical quantities that are vector quantities: the speed with which a mobile travels, since it is not defined only by its module, which is what the speedometer marks, in the case of a car, but it is required to indicate the direction (towards which it is directed), the force acting on an object, since its effect depends in addition to its magnitude or modulus, on the direction in which it acts; also, the displacement of an object, since it is necessary to define the starting and ending point of the movement.

RMSE: root mean square error

The root mean square error (RMSE) is the standard deviation of the residuals (prediction errors). The residuals are a measure of how far the data points are from the regression line; RMSE is a measure of the dispersion of these residues. In other words, it tells you how concentrated the data is around the best-fit line. The root mean square error is commonly used in climatology, prediction, and regression analysis to verify experimental results.
The formula is:

Where:
• f = forecasts (expected values or unknown results),
• o = observed values (known results).

The bar over the squared differences is the mean (similar to x̄). The same formula can be written with the following slightly different notation (Barnston, 1992):

Where:
Σ = summation ("sum")
(z f i - Z o i) Sup> 2 = differences, squared
N = sample size.

You can use whatever formula you feel most comfortable with, as they both do the same thing. If you don't like formulas, you can find the RMSE:
• Squaring the residuals.
• Find the average of the residuals.
• Taking the square root of the result.

That said, this can be a lot of calculation, depending on how big your data is. A shortcut to find the mean square error is:

Where SD y is the standard deviation of Y.
When standardized observations and forecasts are used as RMSE inputs, there is a direct relationship to the correlation coefficient. For example, if the correlation coefficient is 1, the RMSE will be 0, because all the points are on the regression line (and therefore there are no errors)
  
</H5>  

# Research3
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>
 
 <H4>Pipeline and Confusion Matrix</H4>

 <H5 aling="center"> 


Pipeline Definition

Pipeline is an English term that can be translated as "pipeline". Although it is not part of the dictionary of the Royal Spanish Academy
(RAE), the concept is used in our language to refer to a computing architecture.

In Computer Architecture

The pipeline is a technique for implementing instruction-level concurrency within a single processor. Pipelining tries to keep each part of the processor busy, dividing the incoming instructions into a series of sequential steps, which are performed by different processor units that work simultaneously. It increases CPU performance at a certain clock speed, although it can increase latency due to the additional overhead of the pipeline process itself.

In computing and programming

The pipeline architecture (based on filters) consists in transforming a data flow into a process comprised of several sequential phases, the input of each being the output of the previous one. This architecture is very common in the development of programs for the command interpreter, since you can easily connect commands with pipes (pipe).
It is also a very natural architecture in the functional programming paradigm, since it is equivalent to the composition of mathematical functions.

Confusion matrix

In the field of artificial intelligence, a confusion matrix is ​​a tool that allows the visualization of the performance of an algorithm that is used in supervised learning. Each column in the array represents the number of predictions for each class, while each row represents the instances in the actual class. One of the benefits of confusion matrices is that they make it easy to see if the system is confusing two classes.
If the number of samples from different classes changes a lot in the input data, the error rate of the classifier is not representative of how well the classifier performs the task. If for example there are 990 samples from class 1 and only 10 from class 2, the classifier can easily have a bias towards class 1. If the classifier classifies all the samples as class 1, its precision will be 99%. This does not mean that it is a good classifier, as it had a 100% error in classifying class 2 samples.

In the example matrix below, of 8 real cats, the system predicted that three were dogs and out of six dogs predicted that one was a rabbit and two were cats. From the matrix it can be seen that the system has trouble distinguishing between cats and dogs, but that it can reasonably well distinguish between rabbits and other animals. 
 </H5> 
