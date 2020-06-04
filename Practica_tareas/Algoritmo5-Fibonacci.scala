
 
 //5 algoritmos de la secuencia de Fibonacci.
 //Garcia Bautista Ana Laura #15210793
 
 //5.- Versión descendente recursiva 
 // Si el número ingresado en la función es menor que 2, el número ingresado será devuelto
 // si no es menor que 2, la función hará una serie de sumas y el resultado devolverá lo 
 def   función ( n : Int ) :  Int  =  
    {  
    
      si (n < 2 )  
     {  
      volver n  
    } 
   
    contrario  
   
    {
      función de retorno (n - 1 ) + función (n - 2 )  
    }  
    }  
   
    función ( 10 )