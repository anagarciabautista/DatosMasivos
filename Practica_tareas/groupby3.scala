// Enciso Maldonado Aileen Yurely 15210329
// Garcia Bautista Ana Laura 15210793
// método groupBy para agrupar elementos en la secuencia de donas por el primer carácter de cada donut
println("\nStep 2: How to group elements in a sequence using the groupBy function")
val donutsGroup: Map[Char, Seq[String]] = donuts.groupBy(_.charAt(0))
println(s"Group elements in the donut sequence by the first letter of the donut name = $donutsGroup")