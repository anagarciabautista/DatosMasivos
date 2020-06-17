 
<H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>
 
 <H2> Test 3 </H2>
 
 <H5 aling="center">
 
1.- Spark library
 
2.- so that it does not mark errors

3.- we load the csv

4.- select the columns that we are going to train

5.- we train the data that we put in the vectorassembler

6.- we pass the trained data to the model

7.- Evaluate the grouping by calculating Within the sum of quadratic errors.

8.- result </H5>

     import org.apache.spark.sql.SparkSession
     import org.apache.spark.ml.clustering.KMeans
     import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
     import org.apache.spark.ml.linalg.Vectors

     import org.apache.log4j._
     Logger.getLogger("org").setLevel(Level.ERROR)

     val spark = SparkSession.builder().getOrCreate()
     val dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")

    val feature_data = df.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
    val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper",   
    "Delicassen")).setOutputCol("features")

    val traning = assembler.transform(feature_data)

    val kmeans = new KMeans().setK(2).setSeed(1L)      
    val model = kmeans.fit(traning)
    val WSSSE = model.computeCost(traning)
    
    println(s"Within Set Sum of Squared Errors = $WSSSE")
    
    println("Cluster Centers: ")
    
    model.clusterCenters.foreach(println)


