
//Examen unidad 1
//Enciso Maldonado Aileen Yurely 152103289
//Garcia Bautista Ana Laura 15210793
//Datos Masivos
// Variables de la Matriz

  
   val arre = (( 11 , 2 , 4 ), ( 4 , 5 , 6 ), ( 10 , 8 , - 12 ))


def  DAbsoluta (arre : (( Int , Int , Int ), ( Int , Int , Int ), ( Int , Int , Int ))) :  Int  = {
    val  diagonal_1  = (arre._1._1) + (arre._2._2) + (arre._3._3)
    val  diagonal_2  = (arre._1._3) + (arre._2._2) + (arre._3._1)

    var  diferenciacalculada = diagonal_1 - diagonal_2
    var  resul = math.abs (diferenciacalculada)
    return resul
}

DAbsoluta(arre)