
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

<H3> Units Section</H3>  

 <li type="type="square""><a href="https://github.com/anagarciabautista/DatosMasivos/tree/Unidad_1" target="_blank">Unit 1</a></li>
 <ul>
 
 <li type="circle"><a href="https://github.com/anagarciabautista/DatosMasivos/tree/Unidad_1/Evaluacion" target="_blank">
Evaluation</a></li>
 </u>
 
 <H3>Table of Contents</H3>

[1.-Practice 1](#Practice-1) 

[2.-Practice 2](#Practice-2) 

[3.-Fibonacci sequence algorithms 1](#Fibonacci-sequence-algorithms-1) 

[3.1.-Fibonacci sequence algorithms 2](#Fibonacci-sequence-algorithms-2) 

[3.2.-Fibonacci sequence algorithms 3](#Fibonacci-sequence-algorithms-3) 

[3.3.-Fibonacci sequence algorithms 4](#Fibonacci-sequence-algorithms-4) 

[3.4.-Fibonacci sequence algorithms 5](#Fibonacci-sequence-algorithms-5) 

[4.-DF Functions 1](#DF-Functions-1)

[4.1.-DF Functions 2](#DF-Functions-2) 

[4.2.-DF Functions 3](#DF-Functions-3) 

[4.3.-DF Functions 4](#DF-Functions-4) 

[4.4.-DF Functions 5](#DF-Functions-5) 

[4.5.-DF Functions 6](#DF-Functions-6) 

[4.6.-DF Functions 7](#DF-Functions-7) 

[4.7.-DF Functions 8](#DF-Functions-8) 

[5.-RESEARCH WORK 1](#RESEARCH-WORK-5)

[6.-RESEARCH WORK 2](#RESEARCH-WORK-6)

[7.-TEST EXAM 1](#TEST-EXAM-1)

[8.-TEST EXAM 2](#TEST-EXAM-2)

 

<H1 aling="center">
Evaluation 1 </h1>

# PRACTICE 1
  
 <H3 aling="center">

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center">
1. Develop a scale algorithm that calculates the radius of a circle.
 
 2. Develop an algorithm in scala that tells me if a number is a prime number.
 
 3. Given the variable bird = "tweet", use string interpolation to print "I am writing a tweet".
 
 4. Given the variable message = "Hello Luke, I'm your father!" use slilce to extract the "Luke" sequence.
 
 5. What is the difference in value and a variable in scale?
 
 6.Response: value (val) is assigned a defined value and cannot be changed, in a variable
 
 6. Given the tuple (2,4,5,1,2,3,3.1416,23) returns the number 3.1416.</H5>
 
<h2> Code </h2>


    Print ("ingrese el valor del circulo:")
    valor var : Doble scala.oi. StdIn .readLine.toDouble 
    var radio : Double = (valor / ( 2 * 3.1416 ))
    println ( s " \ n \ n el radio del círculo es $ radio " )
    
    def  isprime (num : Int ) : Boolean = {
    if (num <=  1 )
    devuelve  false
    else  if (num == 2 )
    devuelve  true 
    else {
    
    var  i : Int = 0 
    var  bird = " tweet " 
    println ( s " Estoy escribiendo un $ bi rd " )
    var  root : Int = math.sqrt ( num : Double ) .toInt
    for (i < -  2 to root) {
    if (num %i ==  0 ) {
    return  false ;
    }
    }
    volver  verdadero ;
    }
    }
     def  numprime () {
     println ( " Ingrese un número " )
     var  num : Int = scala.io. StdIn .readLine.toInt
     val  Resultado = isprime (num)
     if ( Resultado  ==  verdadero ) {
     println ( s " El número $ num es un número primo " )
      }
      más {
      println ( s " El número $ num no es un número primo " )
      }
      }
      numprime ()
      var  bird = " tweet " 
      println ( s " Estoy escribiendo un $ bi rd " )
      
    var  variable  =  " Hola Luke soy tu padre! "
     var  res = variable.slice ( 5 , 9 )
    println (res)

    var  x  = ( 2 , 4 , 5 , 1 , 2 , 3 , 3.1416 , 23 )
    println (x._7)</H4>

# PRACTICE 2
  
 <H4 aling="center">
 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H4>

<H5 aling="center">

1. Create a list called "list" with the elements "red", "white", "black"

2. Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"

3. Bring the "list" "green", "yellow", "blue" items

4. Create a number array in the 1-1000 range in 5-in-5 steps

5. What are the unique elements of the List list (1,3,3,4,6,7,3,7) use conversion to sets.

6. Create a mutable map called names containing the following "José", 20, "Luis", 24, "Ana", 23, "Susana", "27"

6 a. Print all map keys

6 b. Add the following value to the map ("Miguel", 23)</H5>

<h2> Code </h2>

     import scala.collection.mutable.ListBuffer
     var lista = collection.mutable.ListBuffer("rojo","blanco","negro")
    
    lista += "verde"
    lista += "amarillo"
    lista += "azul"
    lista += "naranja"
    lista += "perla"
    
    lista slice (3,6)

    Array.range(1, 1000, 5)

    lista.toSet

    var names = collection.mutable.Mapa(("Jose", 20), ("Luis", 24), ("Ana", 23) ("Susana", 27))

    nombres.claves

    nombres += ("Miguel" -> 23)
      
  
  <H1 aling="center">5 FIBONACCI ALGORTIMS</H1>
   
 # Fibonacci sequence algorithms 1 
  
  <H3 aling="center">
  Garcia Bautista Ana Laura #15210793
 
  Enciso Maldonado Aileen Enciso #15210329</H3>

 <H5 aling="center">
1.-In this fifth algorithm a function is performed that requests an integer value (Int) then returns an integer value
 
 2.-With decimals (Double) A matrix is ​​created that starts from 0 to (n + 1) if the variable (n) is less than 2, that same
 
 3.-Variable is returned as a result otherwise, the vector with space (0) will have a value of zero (0) and
 
 4.-The vector with space (1) will have a value of one (1) Start pedaling with a for the vector, the
 
 5.-result will be the variable (n) according to the vector of</H5>
 
 <h2> Code </h2>
    
    def función4  ( n : Int ) :  Doble  =
     
     {
    
      val  vector  =  Array .range ( 0 , n +  1 )
      si (n < 2 )
    {
      volver (n)
    }
      mas
    {
      vector ( 0 ) =  0 
      vector ( 1 ) =  1 
   
     para (k < -  2 a n)
    {
     vector (k) = vector (k - 1 ) + vector (k - 2 )
    }
   
     retorno del vector establecido (n)
     } 
    }
    función4 ( 10 )
     
 
 <H1 aling="center">5 FIBONACCI ALGORTIMS</H1>
   
 # Fibonacci sequence algorithms 2.
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
 <H3 aling="center"> 
 2.- Version with explicit formula</H3>
 
 <H5 aling="center"> 
 
If the number entered in the function is less than 2, the number entered will be returned
 if it is not less than 2, the following will be done the formula is divided into parts to finally create the result</H5>
 
 <h2> Code </h2>
  
     def función1 ( n : Doble ) :  Doble  =  
    {  
     si (n < 2 )  
    {
      volver n
    }  
   
    {  
      var  p  = (( 1 + ( Math .sqrt ( 5 ))) / 2 )  
      var  a  =  Math .pow (p, n)  
      var  b  =  Math .pow ( ( 1 - p), n)  
      var  j  = ((a - (b))) / ( Math .sqrt ( 5 ))  
      return (j)  
     }  
    }  
     function1 ( 10 )
      
  <H1 aling="center">5 FIBONACCI ALGORTIMS</H1>
   
  # Fibonacci sequence algorithms 3
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
 <H5 aling="center">
 3.-In this fourth quarter a function was added that after having carried out the corresponding operations,
 the function will give us a result (return) this must be an integer value (Int) A cycle (for) begins where k = 1,
 start cycling until it becomes (n), (n) represents the value that will be entered into the function
 depending on the cycle (for) the variables (b, a) will begin to change their result until the end of the cycle (for)
 the result will be returned with (return)</H5>

 <h2> Code </h2>
 
       def  function3 ( n : Int ) :  Int  = {
         var  a  =  0 
         var  b  =  1 
  
       para (k < -  1 a n)
      { 
         b = b + a
         a = b - a

        }
  
         return (a)
       }
         función3 ( 10 )
       
   
  <H1 aling="center">5 FIBONACCI ALGORTIMS</H1>
   
  # Fibonacci sequence algorithms 4
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
 <H5 aling="center">
 4.-In this sixth algorithm a function was added that after having performed the corresponding operations,
 the function will give us a result (return) this must be an integer value with decimal points (Double)
 if the value entered is less than or equal to 0, then that value will be returned Otherwise, it will have
 to perform a series of operations of starting a cycle (while) where the variables will begin to change the value
 depending on the iteration of the cycle if the variable (i) is odd, different operations will be performed
 If the variable (i) is even, different operations will be made, the variable (i) will begin to change the value each
 Once the cycle is entered until you exit the cycle and the sum of (a + b) is returned</H5>
 
 <h2> Code </h2>
 
     def función5 ( n : Doble ) :  Doble  =
      {

      si (n <=  0 )
       {
         volver (n)
          
       }

         otra persona
           {
            var  i :  Double  = n -  1 
            var  auxOne :  Double  =  0 
            var  auxTwo :  Double  =  1 
            var  a :  Double  = auxTwo
            var  b :  Double  = auxOne
            var  c :  Double  = auxOne
            var  d :  Double  = auxTwo
 
          mientras (i >  0 )
       {

         si  (i %  2  ==  1 )
       {
     
       auxOne = (d * b) + (c * a)
       auxTwo = ((d + (b * a)) + (c * b))
       a = auxOne
       b = auxTwo
     }

        otra cosa
     {
     
      var  pow1  =  Math .pow (c, 2 )
      var  pow2  =  Math .pow (d, 2 )
      auxOne = pow1 + pow2
      auxTwo = (d * (( 2  * (c)) + d))
      c  = auxOne
      d = auxTwo
       } 
      
      i = (i /  2 )
    }
    
    retorno (a + b)
      }
    }
    Función5 ( 9 )


  <H1 aling="center"> 5 FIBONACCI ALGORTIMS</H1>
    
 # Fibonacci sequence algorithms 5
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
 <H5 aling="center">
 5.- Recursive descending version If the number entered in the function is less than 2, the number entered will be returned
 if it is not less than 2, the function will do a series of sums and the result will return</H5> 
 
 <h2> Code </h2>
   
    def   función ( n : Int ) :  Int  =  
    {  
    
      si (n < 2 )  
     {  
      volver n  
    } 
   
    contrario  
   
    {
      función de retorno (n - 1 ) + función (n - 2 )  
    }  
    }  
   
    función ( 10 )
    
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
  
  <H1 aling="center">Practices</H1>
   
  # DF Functions 1
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
  <H5 aling="center">
  assimilate to the SQL clause "GROUP BY", the Spark groupBy () function is used to collect identical data in groups in DataFrame / Dataset and perform aggregate functions in the grouped data. In this article, I will explain groupBy () examples with the Scala language</H5>
  
  <h2> Code </h2>

     import org.apache.spark.sql.SparkSession
    val spark = SparkSession.builder().getOrCreate()
    val df = spark.read.option("header", "true").option("inferSchema","true")csv("Sales.csv")

     //Company, Person, Sales

     //Group By on single column
       df.groupBy("Person").count().show(false)
       df.groupBy("Person").avg("Sales").show(false)
       df.groupBy("Person").sum("Sales").show(false)
       df.groupBy("Person").min("Sales").show(false)
       df.groupBy("Person").max("Sales").show(false)
       df.groupBy("Person").mean("Sales").show(false)

    //GroupBy on multiple columns
     df.groupBy("Company", "Person")
    .sum("Sales","Sales")
    .show(false)
     df.groupBy("department","state")
    .avg("salary","bonus")
    .show(false)
    df.groupBy("department","state")
    .max("salary","bonus")
    .show(false)
    df.groupBy("department","state")
    .min("salary","bonus")
    .show(false)
    df.groupBy("department","state")
    .mean("salary","bonus")
    .show(false)

    //Running Filter
    df.groupBy("department","state")
    .sum("salary","bonus")
    .show(false)

     //using agg function
    df.groupBy("Company")
    .agg(
      sum("Person").as("sum_salary"),
      avg("Person").as("avg_salary"),
      sum("Sales").as("sum_bonus"),
      max("Sales").as("max_bonus"))
    .show(false)

    df.groupBy("Company")
    .agg( sum("Person").as("sum_salary")).show(false)

    df.groupBy("department")
    .agg(
      sum("salary").as("sum_salary"),
      avg("salary").as("avg_salary"),
      sum("bonus").as("sum_bonus"),
      stddev("bonus").as("stddev_bonus"))
    .where(col("sum_bonus") > 50000)
    .show(false)
     }
 
   
  # DF Functions 2
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
  <h2> Code </h2>
   
    import org.apache.spark.sql.SparkSession
    val spark = SparkSession.builder().getOrCreate() 
    val df = spark.read.option("header", "true").option("inferSchema","true")csv("CitiGroup2006_2008")

    //1 print schema
     df.printSchema()

    //2 show the dataset
     df.show()

    //3 show the columns the dataset has
     df.columns

    //4 show the volume data
     df.select("Volume").show()

    //5 show the first record of the dataset
     df.first()

    //6 show the 10 records that head the data set
     df.head(10)

    //7 show interesting facts about the data
     df.describe()

    //8 count the total data the data set has
    df.count()

    //9 order the data
     df.sort()

    //10 show data that is between the conditions
    df.filter($"Close" < 490 && $"low" < 300).show()

     //11 draw the correlation
      df.select(corr("High", "Low")).show()
 
    //12 sum all data of high
    df.select(sum("High")).show()

    // 13 mean of data
    df.select(mean("Low")).show()

    //14 max of data
    df.select(max("High")).show()

    //15 min of data
    df.select(min("Low")).show()

    //16 variance of data
    df.select(variance("Low")).show()

    //17 look for an exact data in the column
      df.filter($"High" === 487.0).show()

    //18 count the values that meet the condition
     df.filter($"High" > 480).count()

    //19 sample for months
     df.select(month(df("Date"))).show()

    //20 sample for years
    df.select(year(df("Date"))).show()
  
  # DF Functions 3
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
 
 <H5 aling="center"> 
 We will use this Spark DataFrame to execute groupBy () in "department" columns and calculate aggregates as a minimum, maximum, average, total salary for each group using the aggregate functions min (), max () and sum () respectively. and finally,
 we will also see how to group and add in multiple columns </H5>

<h2> Code </h2>

    import spark.implicits._
    val simpleData = Seq(("CristianR","Sales","NY",90000,34,10000),
    ("Aileen","Sales","NY",86000,56,20000),
    ("Laura","Sales","CA",81000,30,23000),
    ("Alexis","Finance","CA",90000,24,23000),
    ("Rubensito","Finance","CA",99000,40,24000),
    ("Afedito","Finance","NY",83000,36,19000),
    ("Cynthia","Finance","NY",79000,53,15000),
    ("Irving","Marketing","CA",80000,25,18000),
    ("Ramon","Marketing","NY",91000,50,21000)
     
     )
    
    val df = simpleData.toDF("employee_name","department","salary","state","age","bonus")
    df.show()
    
   
 # DF Functions 3
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
 
  <H5 aling="center"> 
Data Type Verification sequence of donut elements where each element of the sequence is of type String </H5>

<h2> Code </h2>
        
         println("Step 1: How to initialize a Sequence of donuts")
         val donuts: Seq[String] = Seq("Plain Donut", "Strawberry Donut", "Glazed Donut")
         println(s"Elements of donuts = $donuts"
  
 # DF Functions 4
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3> 
  
 <H5 aling="center"> groupBy method to group elements in the donut sequence by the first character of each donut </H5>
 
 <h2> Code </h2>

     println("\nStep 2: How to group elements in a sequence using the groupBy function")
     val donutsGroup: Map[Char, Seq[String]] = donuts.groupBy(_.charAt(0))
     println(s"Group elements in the donut sequence by the first letter of the donut name = $donutsGroup")

 # DF Functions 5
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
<H5 aling="center"> 
Example of class creation with case class object to represent donut objects. </H5>

<h2> Code </h2>

      println("\nStep 3: How to create a case class to represent Donut objects")
      case class Donut(name: String, price: Double)
      
      
  # DF Functions 6
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
<H5 aling="center">
Donut case class from Step 3 and create a sequence of donut elements of type Donut.</H5>

<h2> Code </h2>

     println("\nStep 4: How to create a Sequence of type Donut")
     val donuts2: Seq[Donut] = Seq(Donut("Plain Donut", 1.5), Donut("Strawberry Donut", 2.0), Donut("Glazed Donut", 2.5))
     println(s"Elements of donuts2 = $donuts2")

   
  # DF Functions 7
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
 
<H5 aling="center">
 group the donut objects represented by the Donut case class from Step 3 by their name property using the groupBy method</H5>
 
 <h2> Code </h2>

     println(s"\nStep 5: How to group case classes donut objects by the name property")
     val donutsGroup2: Map[String, Seq[Donut]] = donuts2.groupBy(_.name)
     println(s"Group element in the sequence of type Donut grouped by the donut name = $donutsGroup2"

  # DF Functions 8
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
  <h2> Code </h2>
 
    import org.apache.spark.sql.SparkSession
    val spark = SparkSession.builder().getOrCreate()
    val df = spark.read.option("header", "true").option("inferSchema","true")csv("CitiGroup2006_2008")

    //1.sumDistinct
    df.select(sumDistinct("Sales")).show()

     //2.last
    df.select(last("Company")).show() //last data in company

    //3.first
    df.select(first("Person")).show() first data in person

    //4.var_pop
    df.select(var_pop("Sales")).show()

    //5.avg
    df.select(avg("Sales")).show()

    //6.collect_list
    df.select(collect_list("Sales")).show()

    //7.var_samp
    df.select(var_samp("Sales")).show()

    //8.sum
    df.select(sum("Sales")).show()

    //9.stddev_pop
    df.select(stddev_pop("Sales")).show()

    //10.skewness
    df.select(skewness("Sales")).show()

    //11.min
     df.select(min("Sales")).show()

     //12.kurtosis
     df.select(kurtosis("Sales")).show()

    //13.collect_set
     df.select(collect_set("Sales")).show()

    //14.approx_count_distinct
    df.select(approx_count_distinct("Company")).show()

    //15.mean
    df.select(mean("Sales")).show()

    //16 return the first column of the dataframe
      df.first 

     //17 Returns the dataframe columns
       df.columns 

     //18 Add a column that derives from the high and Volume column
      val df2 = df.withColumn("HV Ratio", df("High")+df("Volume")) 

    //19 Choose the volume column min
     df.select(min("Volume")).show() 

    //20 Choose the volume column max
    df.select(max("Volume")).show() 

 # RESEARCH WORK 1
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
  <H3 aling="center"> Variance</H3>
   
 <H3 aling="center">
 What Is Variance?</H3> 

<H5 aling="center">
 Variance (σ2) in statistics is a measurement of the spread between numbers in a data set. That is, it measures how far each number in   the set is from the mean and therefore from every other number in the set. In investing, the variance of the returns among assets in   a portfolio is analyzed as a means of achieving the best asset allocation. The variance equation, in financial terms, is a formula    for comparing the performance of the elements of a portfolio against each other and against the mean.</H5> 

<H3 aling="center">Understanding Variance </H3>

<H5 aling="center">
 Variance is calculated by taking the differences between each number in the data set and the mean, then squaring the differences to make them positive, and finally dividing the sum of the squares by the number of values in the data set.</H5>

<H3 aling="center">How to Use Variance </H3>

<H5 aling="center">
 Variance measures variability from the average or mean. To investors, variability is volatility, and volatility is a measure of risk.  Therefore, the variance statistic can help determine the risk an investor assumes when purchasing a specific security. 
A large variance indicates that numbers in the set are far from the mean and from each other, while a small variance indicates the opposite. Variance can be negative. A variance value of zero indicates that all values within a set of numbers are identical.</H5>
 
 <center>
 <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcReADNKws5_F-VLI5tn5b70h3jeb24nj6nntIp67k2iH-iMvSUY" width="350"
       height="200">
</center>
   
   
  # RESEARCH WORK 2
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
  <H3 aling="center"> Person Correlation</H3>
   
 <H3 aling="center">
 What does this test do?</H3>
 
<H5 aling="center">
The Pearson product-moment correlation coefficient (or Pearson correlation coefficient, for short) is a measure of the strength of a linear association between two variables and is denoted by r. Basically, a Pearson product-moment correlation attempts to draw a line of best fit through the data of two variables, and the Pearson correlation coefficient, r, indicates how far away all these data points are to this line of best fit (i.e., how well the data points fit this new model/line of best fit).</H5>

<H3 aling="center">What values can the Pearson correlation coefficient take?</H3>

<H5 aling="center">
The Pearson correlation coefficient, r , can take a range of values from +1 to -1. A value of 0 indicates that there is no association between the two variables. A value greater than 0 indicates a positive association; that is, as the value of one variable increases, so does the value of the other variable. A value less than 0 indicates a negative association; that is, as the value of one variable increases, the value of the other variable decreases. This is shown in the diagram below:</H5>

<center>
 <img src="https://statistics.laerd.com/statistical-guides/img/pc/pearson-1-small.png" width="600"
       height="300">
</center>

<H3 aling="center">How can we determine the strength of association based on the Pearson correlation coefficient?</H3>

<H5 aling="center">
 The stronger the association of the two variables, the closer the Pearson correlation coefficient, r, will be to either +1 or -1 depending on whether the relationship is positive or negative, respectively. Achieving a value of +1 or -1 means that all your data points are included on the line of best fit – there are no data points that show any variation away from this line. Values for r between +1 and -1 (for example, r = 0.8 or -0.4) indicate that there is variation around the line of best fit. The closer the value of r to 0 the greater the variation around the line of best fit. Different relationships and their correlation coefficients are shown in the diagram below:</H5>
 
 <center>
 <img src="https://statistics.laerd.com/statistical-guides/img/pc/pearson-2-small.png" width="600"
       height="400">
</center>

 

  
