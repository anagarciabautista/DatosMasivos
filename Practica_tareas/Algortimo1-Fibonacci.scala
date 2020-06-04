 //5 algoritmos de la secuencia de Fibonacci.
 //Garcia Bautista Ana Laura #15210793

 //1.-En este quinto algoritmo se realiza una función que solicita un valor entero (Int) luego devuelve un valor entero 
 //con decimales (Doble) Se crea una matriz que comienza de 0 a (n + 1) si la variable (n) es menor que 2, esa misma 
 ///variable se devuelve como resultado de lo contrario, el vector con espacio (0) tendrá un valor de cero (0) y 
 //el vector con espacio (1) tendrá un valor de uno (1) Comience a pedalear con a para el vector, el 
 //resultado será la variable (n) de acuerdo con el vector de
 def función4  ( n : Int ) :  Doble  =
    {
    
      val  vector  =  Array .range ( 0 , n +  1 )
      si (n < 2 )
    {
      volver (n)
    }
      mas
    {
      vector ( 0 ) =  0 
      vector ( 1 ) =  1 
   
     para (k < -  2 a n)
    {
     vector (k) = vector (k - 1 ) + vector (k - 2 )
    }
   
     retorno del vector establecido (n)
     } 
    }
    función4 ( 10 )
   