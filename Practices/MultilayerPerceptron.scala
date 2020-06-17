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
11.- Se imprime el error de la precision 

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