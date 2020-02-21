 // 5 algoritmos de la secuencia de Fibonacci.
 //Garcia Bautista Ana Laura #15210793
 
 //2.- Versión con fórmula explícita
 // Si el número ingresado en la función es menor que 2, el número ingresado será devuelto
 // si no es menor que 2, se hará lo siguiente la fórmula se divide en partes para finalmente crear el resultado
  def función1 ( n : Doble ) :  Doble  =  
    {  
     si (n < 2 )  
    {
      volver n
    }  
   
    {  
      var  p  = (( 1 + ( Math .sqrt ( 5 ))) / 2 )  
      var  a  =  Math .pow (p, n)  
      var  b  =  Math .pow ( ( 1 - p), n)  
      var  j  = ((a - (b))) / ( Math .sqrt ( 5 ))  
      return (j)  
     }  
    }  
     function1 ( 10 )