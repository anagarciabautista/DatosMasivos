
//Examen unidad 1
//Enciso Maldonado Aileen Yurely 152103289
//Garcia Bautista Ana Laura 15210793
//Datos Masivos
// Variables de la Matriz

 val arr = ((11,2,4),(4,5,6),(10,8,-12))
 //verificacion de ariables enteras
 def diagonaldifference(arr2:((Int, Int, Int), (Int, Int, Int), (Int, Int, Int))): 
 Int = {
    //Se inica el arreglo de la matriz
    val diagonal_1 =(arr._1._1)+(arr._2._2)+(arr._3._3)
    val diagonal_2 =(arr._1._3)+(arr._2._2)+(arr._3._1)
    //Direfencia absoluta de la matriz
    var diferencia = diagonal_1-diagonal_2
    var resulado= math.abs(diferenciaabsoluta)
    //Regresa el rsultado de un valor entero de la matriz
    return resultado
}
diagonaldifference(arr)