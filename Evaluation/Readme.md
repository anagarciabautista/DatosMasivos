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
<H4 aling="center">Unit 3</H4>
<H3 aling="center">Student's name</H3>
<H4 aling="center">
 Garcia Bautista Ana Laura # 15210793 

 Enciso Maldonado Aileen Yurely #15210329</H4> 

<H3 aling="center">Teacher (A):</H3>
<H4 aling="center">Romero Hernández José Christian </H4>

<H3> Units Section</H3>  

 <li type="type="square""><a href="https://github.com/anagarciabautista/DatosMasivos/tree/Unidad_3" target="_blank">Unit 3/a></li>
 <ul>
 
 <li type="circle"><a href="https://github.com/anagarciabautista/DatosMasivos/tree/Unidad_3/Evaluation" target="_blank">
Evaluation</a></li>
 </u>
 
<H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>
 
 <H2> Test </H2>
 
 <H5 aling="center">
1.- Libreria spark
2.- para que no marque errores 
3.- cargamos el csv 
4.- selecionamos las columnas que vamos a entrenar
5.- entrenamos la data que colocamos en el vectorassembler
6.- la data entrenada la pasamos al modelo 
7.- Evaluate clustering by calculate Within Set Sum of Squared Errors.
8.- resultado </H5>

     import org.apache.spark.sql.SparkSession
     import org.apache.spark.ml.clustering.KMeans
     import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
     import org.apache.spark.ml.linalg.Vectors

     import org.apache.log4j._
     Logger.getLogger("org").setLevel(Level.ERROR)

     val spark = SparkSession.builder().getOrCreate()
     val dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")

    val feature_data = df.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
    val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

    val traning = assembler.transform(feature_data)

    val kmeans = new KMeans().setK(2).setSeed(1L)      
    val model = kmeans.fit(traning)
    val WSSSE = model.computeCost(traning)
    
    println(s"Within Set Sum of Squared Errors = $WSSSE")
    
    println("Cluster Centers: ")
    
    model.clusterCenters.foreach(println)


