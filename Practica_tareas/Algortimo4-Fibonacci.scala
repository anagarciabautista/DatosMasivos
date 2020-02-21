//5 algoritmos de la secuencia de Fibonacci.
 //Garcia Bautista Ana Laura #15210793

 
 //4.-En este sexto algoritmo se agregó una función que después de haber realizado las operaciones correspondientes, 
 //la función nos dará un resultado (retorno) este debe ser un valor entero con puntos decimales (Doble)
 //si el valor ingresado es menor o igual a 0, entonces ese valor se devolverá De lo contrario, tendrá 
 //que realizar una serie de operaciones de iniciar un ciclo ( while) donde las variables comenzarán a cambiar el valor 
 //dependiendo de la iteración del ciclo si la variable (i) es impar, se realizarán diferentes operaciones
 // Si la variable (i) es par, diferentes operaciones se harán la variable (i) comenzará a cambiar el valor cada
 // vez que se ingrese el ciclo hasta que salga del ciclo y se devuelva la suma de (a + b) 
  def función5 ( n : Doble ) :  Doble  =
 {

  si (n <=  0 )
 {
   volver (n)
 }

 otra persona
  {
    var  i :  Double  = n -  1 
    var  auxOne :  Double  =  0 
    var  auxTwo :  Double  =  1 
    var  a :  Double  = auxTwo
    var  b :  Double  = auxOne
    var  c :  Double  = auxOne
    var  d :  Double  = auxTwo
 
   mientras (i >  0 )
 {

   si (i %  2  ==  1 )
 {
   auxOne = (d * b) + (c * a)
   auxTwo = ((d + (b * a)) + (c * b))
   a = auxOne
   b = auxTwo
 }

  otra cosa
 {
   var  pow1  =  Math .pow (c, 2 )
   var  pow2  =  Math .pow (d, 2 )
   auxOne = pow1 + pow2
   auxTwo = (d * (( 2  * (c)) + d))
   c = auxOne
   d = auxTwo
 } 
   i = (i /  2 )
 }
   retorno (a + b)
 }
}
 Función5 ( 9 )
