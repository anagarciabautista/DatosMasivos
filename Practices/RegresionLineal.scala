
import org.apache.spark.ml.regression.LinearRegression

// Cargar datos de entrenamiento
val training =spark.read.format("libsvm").load("data/mllib/sample_linear_regression_data.txt")
val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

// Ajustar el modelo
val lrModel = lr.fit(training)

// Imprime los coeficientes e intercepta para regresión lineal
println(s"Coefficients: ${lrModel.coefficients} Intercept:${lrModel.intercept}")

// Resuma el modelo sobre el conjunto de entrenamiento e imprima algunas métricas
val trainingSummary = lrModel.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory:[${trainingSummary.objectiveHistory.mkString(",")}]")
trainingSummary.residuals.show()println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")