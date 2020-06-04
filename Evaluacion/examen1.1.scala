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

