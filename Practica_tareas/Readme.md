<H1 aling="center"> 
Tecnológico Nacional De México Instituto Tecnológico De Tijuana
Subdirección Académica
Departamento De Sistemas Y Computación</H1>

<H1 aling="center">SEMESTRE:</H1>
<H3 aling="center">Enero – junio 2020</H3>

<H1 aling="center">CARRERA:</H1>
<H3 aling="center">Ing. Tecnologías De Información Y Comunicación</H3>

<H1 aling="center">MATERIA:</H1>
<H3 aling="center">Datos Masivos</H3>

<H1 aling="center">UNIDAD A EVALUAR:</H1>
<H3 aling="center">Unidad 1</H3>

<H1 aling="center">NOMBRE DEL ALUMNO:</H1>
<H3 aling="center">Garcia Bautista Ana Laura # 15210793
Enciso Maldonado Aileen Yurely #15210329</H3> 

<H1 aling="center">MAESTRO (A):</H1>
<H3 aling="center">Romero Hernández José Christian </H3>


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

<H4 aling="center">
print ( " Ingrese el valor del círculo: " )
 valor var  : Doble = scala.io. StdIn .readLine.toDouble
 var radio : Double = (valor / ( 2 * 3.1416 ))   
println ( s " \ n \ n el radio del círculo es $ radio " )</h4>
 

<H4 aling="center">
def  isprime (num : Int ) : 
 Boolean = {
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
}</H4>


<H4 aling="center">
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
}</H4>


<H4 aling="center">
numprime ()

    var  bird = " tweet " 
    println ( s " Estoy escribiendo un $ bi rd " )

    var  variable  =  " Hola Luke soy tu padre! "
     var  res = variable.slice ( 5 , 9 )
    println (res)

    var  x  = ( 2 , 4 , 5 , 1 , 2 , 3 , 3.1416 , 23 )
    println (x._7)</H4>

<H1 aling="center">
 Práctica 2
 Garcia Bautista Ana Laura #15210793
 
 Enciso Maldonado Aileen Yurely #15210329

 1. Crea una lista llamad "lista" con los elementos "rojo", "blanco", "negro"
 2. Añadir 5 elementos mas a "lista" "verde", "amarillo", "azul", "naranja", "perla"
 3. Traer los elementos de "lista" "verde", "amarillo", "azul"
 4. Crea un arreglo de número en rango del 1-1000 en pasos de 5 en 5
 5. Cuales son los elementos unicos de la lista Lista (1,3,3,4,6,7,3,7) usan conversion a conjuntos
 6. Crea un mapa mutable llamado nombres que contenga los siguientes "José", 20, "Luis", 24, "Ana", 23, "Susana", "27"
 6 a. Imprime todas las llaves del mapa
 7 b. Agrega el siguiente valor al mapa ("Miguel", 23)</H1>

<H4 aling="center">
import scala.collection.mutable.ListBuffer
var lista = collection.mutable.ListBuffer("rojo","blanco","negro")</H4>

<H4 aling="center">
lista += "verde"
lista += "amarillo"
lista += "azul"
lista += "naranja"
lista += "perla"</H4>

<H4 aling="center">lista slice (3,6)</H4>

<H4 aling="center">Array.range(1, 1000, 5)</H4>

<H4 aling="center">lista.toSet</H4>

<H4 aling="center">var names = collection.mutable.Mapa(("Jose", 20), ("Luis", 24), ("Ana", 23) ("Susana", 27))</H4>

<H4 aling="center">nombres.claves</H4>

<H4 aling="center">nombres += ("Miguel" -> 23)</H4>



