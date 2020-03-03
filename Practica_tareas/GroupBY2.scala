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