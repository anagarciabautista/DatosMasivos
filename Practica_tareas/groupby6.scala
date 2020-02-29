// Enciso Maldonado Aileen Yurely 15210329
// Garcia Bautista Ana Laura 15210793
//grupar los objetos donut representados por la clase de caso Donut del Paso 3 por su propiedad de nombre usando el método groupBy .
println(s"\nStep 5: How to group case classes donut objects by the name property")
val donutsGroup2: Map[String, Seq[Donut]] = donuts2.groupBy(_.name)
println(s"Group element in the sequence of type Donut grouped by the donut name = $donutsGroup2"