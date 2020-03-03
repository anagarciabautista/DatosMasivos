<H1 aling="center"> 
Tecnológico Nacional De México Instituto Tecnológico De Tijuana
Subdirección Académica
Departamento De Sistemas Y Computación</H1>

<H2 aling="center">SEMESTRE:</H2>
<H3 aling="center">Enero – junio 2020</H3>

<H2 aling="center">CARRERA:</H2>
<H3 aling="center">Ing. Tecnologías De Información Y Comunicación</H3>

<H2 aling="center">MATERIA:</H2>
<H3 aling="center">Datos Masivos</H3>

<H2 aling="center">UNIDAD A EVALUAR:</H2>
<H3 aling="center">Unidad 1</H3>

<H2 aling="center">NOMBRE DEL ALUMNO:</H2>
<H3 aling="center">Garcia Bautista Ana Laura # 15210793
Enciso Maldonado Aileen Yurely #15210329</H3> 

<H2 aling="center">MAESTRO (A):</H2>
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

