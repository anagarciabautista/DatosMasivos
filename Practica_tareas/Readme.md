
<H1 aling="center"> 
Tecnológico Nacional De México Instituto Tecnológico De Tijuana
Subdirección Académica
Departamento De Sistemas Y Computación</H1>

<H3 aling="center">SEMESTRE:</H3>
<H4 aling="center">Enero – junio 2020</H4>

<H3 aling="center">CARRERA:</H3>
<H4 aling="center">Ing. Tecnologías De Información Y Comunicación</H4>

<H3 aling="center">MATERIA:</H3>
<H4 aling="center">Datos Masivos</H4>

<H3 aling="center">UNIDAD A EVALUAR:</H3>
<H4 aling="center">Unidad 1</H4>

<H3 aling="center">NOMBRE DEL ALUMNO:</H3>
<H4 aling="center">
 
 Garcia Bautista Ana Laura # 15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H4> 

<H3 aling="center">MAESTRO (A):</H3>
<H4 aling="center">Romero Hernández José Christian </H4>

<H3 aling="center> Index </h3>
           
 <li type="type="square""><a href="https://github.com/anagarciabautista/DatosMasivos/tree/Unidad_1" target="_blank">Unidad 1</a></li>
   <ul>
  
  <li type="circle"><a href="https://github.com/anagarciabautista/DatosMasivos/tree/Unidad_1/Practica_tareas" target="_blank">Practica-   Tareas</a>  </li>
  <li type="circle"><a href="https://github.com/anagarciabautista/DatosMasivos/tree/Unidad_1/Evaluacion" target="_blank">Evaluacion</a></li>
 
 </u>

<H1 aling="center">
 Evaluación 1 </h1>

 <H3 aling="center">Práctica 1

 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329</H3>

 <H5 aling="center">
 1. Desarrolla un algoritmo en escala que calcula el radio de un círculo.
 
 2. Desarrolle un algoritmo en scala que me diga si un número es un número primo.
 
 3. Dada la variable bird = "tweet", use la interpolación de cadena para imprimir "Estoy escribiendo   un tweet".
 
 4. Dada la variable mensaje = "Hola Luke soy tu padre!" usa slilce para extraer la secuencia "Luke".
 
 5. Cual es la diferencia en valor y una variable en escala?
 
 6.Respuesta :value(val) se le asigna un valor definido y no puede ser cambiado, en una variable
 
 6. Dada la tupla (2,4,5,1,2,3,3.1416,23) devuelve el número 3.1416.</H5>
 
<h1> Codigo </h1>


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

 <H3 aling="center">Práctica 2</H3>
 
 <H4 aling="center">
  Garcia Bautista Ana Laura #15210793
 
  Enciso Maldonado Aileen Yurely #15210329</H4>

<H5 aling="center">
1. Crea una lista llamad "lista" con los elementos "rojo", "blanco", "negro"

2. Añadir 5 elementos mas a "lista" "verde", "amarillo", "azul", "naranja", "perla"
 
3. Traer los elementos de "lista" "verde", "amarillo", "azul"

4. Crea un arreglo de número en rango del 1-1000 en pasos de 5 en 5

5. Cuales son los elementos unicos de la lista Lista (1,3,3,4,6,7,3,7) usan conversion a conjuntos.

6. Crea un mapa mutable llamado nombres que contenga los siguientes "José", 20, "Luis", 24, "Ana", 23, "Susana", "27"

6 a. Imprime todas las llaves del mapa

6 b. Agrega el siguiente valor al mapa ("Miguel", 23)</H5>

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
      
  
  <H1 aling="center">AlGORTIMOS DE FIBONACCI</H1>
   
  <H2 aling="center"> 
  5 algoritmos de la secuencia de Fibonacci.</H2>
 
  <H3 aling="center">
  Garcia Bautista Ana Laura #15210793
 
  Enciso Maldonado Aileen Enciso #15210329</H3>

 <H5 aling="center">
 1.-En este quinto algoritmo se realiza una función que solicita un valor entero (Int) luego devuelve un valor entero 
 
 2.-Con decimales (Doble) Se crea una matriz que comienza de 0 a (n + 1) si la variable (n) es menor que 2, esa misma 
 
 3.-Variable se devuelve como resultado de lo contrario, el vector con espacio (0) tendrá un valor de cero (0) y 
 
 4.-El vector con espacio (1) tendrá un valor de uno (1) Comience a pedalear con a para el vector, el 
 
 5.-resultado será la variable (n) de acuerdo con el vector de</H5>
    
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
     
 
 <H1 aling="center">AlGORTIMOS DE FIBONACCI</H1>
   
  <H2 aling="center"> 
  5 algoritmos de la secuencia de Fibonacci.</H2>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
 <H3 aling="center"> 
 2.- Versión con fórmula explícita</H3>
 
 <H5 aling="center"> 
 Si el número ingresado en la función es menor que 2, el número ingresado será devuelto
 si no es menor que 2, se hará lo siguiente la fórmula se divide en partes para finalmente crear el resultado</H5>
  
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
      
  <H1 aling="center">AlGORTIMOS DE FIBONACCI</H1>
   
  <H2 aling="center"> 
  5 algoritmos de la secuencia de Fibonacci.</H2>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
 <H5 aling="center">
 3.-En este cuarto trimestre se agregó una función que después de haber realizado las operaciones correspondientes, 
 la función nos dará un resultado (retorno) este debe ser un valor entero (Int) Un ciclo (para) comienza donde k = 1,
 comenzará a ciclar hasta que se convierta en (n), (n) representa el valor que se ingresará en la función 
 dependiendo del ciclo (para) las variables (b, a) comenzarán a cambiar su resultado hasta el final del ciclo (para) 
 el resultado se devolverá con (return)</H5>


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
       
   
  <H1 aling="center">AlGORTIMOS DE FIBONACCI</H1>
   
  <H2 aling="center"> 
  5 algoritmos de la secuencia de Fibonacci.</H2>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
 <H5 aling="center">
 4.-En este sexto algoritmo se agregó una función que después de haber realizado las operaciones correspondientes, 
 la función nos dará un resultado (retorno) este debe ser un valor entero con puntos decimales (Doble)
 si el valor ingresado es menor o igual a 0, entonces ese valor se devolverá De lo contrario, tendrá 
 que realizar una serie de operaciones de iniciar un ciclo ( while) donde las variables comenzarán a cambiar el valor 
 dependiendo de la iteración del ciclo si la variable (i) es impar, se realizarán diferentes operaciones
 Si la variable (i) es par, diferentes operaciones se harán la variable (i) comenzará a cambiar el valor cada
 vez que se ingrese el ciclo hasta que salga del ciclo y se devuelva la suma de (a + b)</H5>
 
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


  <H1 aling="center">AlGORTIMOS DE FIBONACCI</H1>
   
  <H2 aling="center"> 
  5 algoritmos de la secuencia de Fibonacci.</H2>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
 <H5 aling="center">
 5.- Versión descendente recursiva 
 Si el número ingresado en la función es menor que 2, el número ingresado será devuelto
 si no es menor que 2, la función hará una serie de sumas y el resultado devolverá </H5> 
   
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
    
  <H1 aling="center">EXAMEN PRUEBA</H1>
   
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
    //Variables de la Matriz
    val arr = ((11,2,4),(4,5,6),(10,8,-12))
 
    //Verificacion de ariables enteras
    def diagonaldifference(arr2:((Int, Int, Int), (Int, Int, Int), (Int, Int, Int))): 
    Int = {
 
    //Se inica el arreglo de la matriz
    val diagonl_1 =(arr._1._1)+(arr._2._2)+(arr._3._3)
    val diagonl_2 =(arr._1._3)+(arr._2._2)+(arr._3._1)
 
    //Direfencia absoluta de la matriz
    var diferencia = diagonal_1-diagonal_2
    var resulado= math.abs(diferenciaabsoluta)
 
    //Regresa el rsultado de un valor entero de la matriz
    return resultado
    }
    diagonaldifference(arr)
  
  <H1 aling="center">Practicas</H1>
   
  <H2 aling="center"> 
  Funciones De DF.</H2>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
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
 
 
 <H1 aling="center">Practicas</H1>
   
  <H2 aling="center"> 
  Funciones De DF.</H2>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
   
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
 
  <H1 aling="center">Practicas</H1>
   
  <H2 aling="center"> 
  Funciones De DF.</H2>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
 
 <H5 aling="center"> 
 Usaremos este Spark DataFrame para ejecutar groupBy () en columnas de "departamento" y calcular agregados como mínimo, máximo,          promedio, salario total para cada grupo usando las funciones de agregado min (), max () y sum () respectivamente. y finalmente,
 también veremos cómo agrupar y agregar en múltiples columnas</H5>

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

 <H1 aling="center">Practicas</H1>
   
  <H2 aling="center"> 
  Funciones De DF.</H2>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
 
  <H5 aling="center"> Verificación de tipo de dato
  Secuencia de elementos donut donde cada elemento de la secuencia es de tipo String</H5>

         println("Step 1: How to initialize a Sequence of donuts")
         val donuts: Seq[String] = Seq("Plain Donut", "Strawberry Donut", "Glazed Donut")
         println(s"Elements of donuts = $donuts"
 
  <H2 aling="center"> 
  Funciones De DF.</H2>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3> 
  
 <H5 aling="center"> método groupBy para agrupar elementos en la secuencia de donas por el primer carácter de cada donut</H5>

     println("\nStep 2: How to group elements in a sequence using the groupBy function")
     val donutsGroup: Map[Char, Seq[String]] = donuts.groupBy(_.charAt(0))
     println(s"Group elements in the donut sequence by the first letter of the donut name = $donutsGroup")

 <H2 aling="center"> 
  Funciones De DF.</H2>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
<H5 aling="center"> Ejemplo de creación de clase con objeto
clase de caso para representar objetos donut.</H5>

      println("\nStep 3: How to create a case class to represent Donut objects")
      case class Donut(name: String, price: Double)
      
  <H2 aling="center"> 
  Funciones De DF.</H2>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
<H5 aling="center">
Clase de caso Donut del Paso 3 y crear una secuencia de elementos donut de tipo Donut.</H5>

    println("\nStep 4: How to create a Sequence of type Donut")
    val donuts2: Seq[Donut] = Seq(Donut("Plain Donut", 1.5), Donut("Strawberry Donut", 2.0), Donut("Glazed Donut", 2.5))
    println(s"Elements of donuts2 = $donuts2")

  <H2 aling="center"> 
  Funciones De DF.</H2>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
 
 
<H5 aling="center">
 agrupar los objetos donut representados por la clase de caso Donut del Paso 3 por su propiedad de nombre usando el método groupBy</H5> 

     println(s"\nStep 5: How to group case classes donut objects by the name property")
     val donutsGroup2: Map[String, Seq[Donut]] = donuts2.groupBy(_.name)
     println(s"Group element in the sequence of type Donut grouped by the donut name = $donutsGroup2"

  <H2 aling="center"> 
  Funciones De DF.</H2>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
 
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

 <H1 aling="center"> 
  TRABAJOS DE INVESTIGACION.</H1>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
  <H3 aling="center"> Investigacion 1</H3>
   
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
   
  
  <H1 aling="center"> 
  TRABAJOS DE INVESTIGACION.</H1>
 
  <H3 aling="center"> 
  Garcia Bautista Ana Laura #15210793
  
  Enciso Maldonado Aileen Yurely #15210329</H3>
  
  <H3 aling="center"> Investigacion 2</H3>
   
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

 

  
