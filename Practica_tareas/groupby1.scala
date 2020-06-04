// Enciso Maldonado Aileen Yurely 15210329
// Garcia Bautista Ana Laura 15210793
//Usaremos este Spark DataFrame para ejecutar groupBy () en columnas de "departamento" y calcular agregados como mínimo, máximo, promedio, salario total para cada grupo usando las funciones de agregado min (), max () y sum () respectivamente. y finalmente, también veremos cómo agrupar y agregar en múltiples columnas

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
