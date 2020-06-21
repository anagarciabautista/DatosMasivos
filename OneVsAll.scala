// Se importan todas la librerias necesarias
//Se agregan las librerias del algortimo
// Se declararan 2 arreglos, uno tendra los datos de entrenamiento y el otro tendra
// los datos de prueba, respectivamente fueron declarados como arreglos y tendran el 80 y 20 porciento de los datos totales
// Se declara la variable "classifier" que hara la regresion
// Se declara el modelo "OneVsRest"
// Se entrena el modelo con los datos de entrenamiento
// Se hacen las predicciones con los datos de prueba
// Se declara el evaluador que tomara la precision del modelo y lo guardara en una variable metrica llamada "accuracy"
// Se calcula el error del modelo con una simple resta

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