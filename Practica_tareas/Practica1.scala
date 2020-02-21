// Evaluación 1 
// Práctica 1
//Garcia Bautista Ana Laura #15210793

// 1. Desarrolla un algoritmo en escala que calcula el radio de un círculo.

print ( " Ingrese el valor del círculo: " )
 valor var  : Doble = scala.io. StdIn .readLine.toDouble
 var radio : Double = (valor / ( 2 * 3.1416 ))   
println ( s " \ n \ n el radio del círculo es $ radio " )


// 2. Desarrolle un algoritmo en scala que me diga si un número es un número primo.

def  isprime (num : Int ) : Boolean = {
 if (num <=  1 )
 devuelve  false 
else  if (num == 2 )
 devuelve  true 
else {
     var  i : Int = 0 
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

// 3. Dada la variable bird = "tweet", use la interpolación de cadena para imprimir "Estoy escribiendo un tweet".

    var  bird = " tweet " 
    println ( s " Estoy escribiendo un $ bi rd " )


// 4. Dada la variable mensaje = "Hola Luke soy tu padre!" usa slilce para extraer la secuencia "Luke".

    var  variable  =  " Hola Luke soy tu padre! "
     var  res = variable.slice ( 5 , 9 )
    println (res)


// 5. Cual es la diferencia en valor y una variable en escala?
//Respuesta :value(val) se le asigna un valor definido y no puede ser cambiado, en una variable

// 6. Dada la tupla (2,4,5,1,2,3,3.1416,23) devuelve el número 3.1416.

    var  x  = ( 2 , 4 , 5 , 1 , 2 , 3 , 3.1416 , 23 )
    println (x._7)



