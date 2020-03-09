# Big Data

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


<H5> Units Section</H5>
           
 <li type="type="square""><a href="https://github.com/anagarciabautista/DatosMasivos/tree/Unidad_1" target="_blank">Unit 1</a></li>
   <ul>
  
  <li type="circle"><a href="https://github.com/anagarciabautista/DatosMasivos/tree/Unidad_1/Practica_tareas" target="_blank">
Practice and Tasks</a>  </li>
 
 </u>
 

<H3>Table of Contents</H3>

[1.-TEST EXAM 1](#TEST-EXAM-1)

[2.-TEST EXAM 2](#TEST-EXAM-2)


 # TEST EXAM 1
   
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
  <h2> Code </h2>
  
    //Matrix Variables
    val arr = ((11,2,4),(4,5,6),(10,8,-12))
 
    //Integer variables verification
    def diagonaldifference(arr2:((Int, Int, Int), (Int, Int, Int), (Int, Int, Int))): 
    Int = {
 
    //Array is started
    val diagonl_1 =(arr._1._1)+(arr._2._2)+(arr._3._3)
    val diagonl_2 =(arr._1._3)+(arr._2._2)+(arr._3._1)
 
    //Absolute leadership of the matrix
    var diferencia = diagonal_1-diagonal_2
    var resulado= math.abs(diferenciaabsoluta)
 
    //Returns the result of an integer value of the array

    return resultado
    }
    diagonaldifference(arr)
    
 # TEST EXAM 2
   
<H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
 <h2> Code </h2>
 
    // punto 1 Comienza una simple sesion Spark
       import org.apache.spark.sql.SparkSession 
       val spark = SparkSession.builder().getOrCreate()

    // punto 2 Cargue el archivo nrtflix stock CSV, haga que apark infiera los tipos de datos
       val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv") 

     // punto 3 Cuales son los nombres de las colunmas date, apen, high, low, close
        df.columns

     // punto 4 como es el esquema
     df.printSchema() 

    // punto 5 imprime las primeras 5 columas
      df.select("Date","Open","High","Low","Close").show()

    // punto 6 Usa describe () para aprender sobre el DataFrame
       df.describe().show()

    // punto 7 Crea un nuevo dataframe con una columna nueva llamada "HV Ratio" que es la relacion entre el precio 
       val df2 = df.withColumn("HV Ratio", df("High")/df("Volume")).show()
 
       // 8
      val df5 = df.withColumn("Day", dayofmonth(df("Date")))
      val df5n = df5.select($"Day", $"Close")
      val df5max= (df5n.select(max("Close")))
      val closemax=df5max.first().getDouble(0)
      df5n.filter($"Close"=== closemax).show()

     // punto 9
     // la columna Close significa como cerro ese dia la bolsa

     // punto 10
     df.select(max("Volume")).show()

     df.select(min("Volume")).show()

    //punto 11
    //A
    df.filter($"Close" < 600).count()

    //B
     df.filter($"high">500).count() *1.0/df.count()*100

    //C
    var total = df.select(corr($"high",$"Volume")).show()

    //d Cual es el maximo de la columa high por ano
      df.groupBy(year(df("Date"))).max().show()

    //e Cual es el promedio de la columna close para cada  mes del calendario
     df.groupBy(month(df("Date"))).avg().show()

