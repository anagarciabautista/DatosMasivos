  //Garcia Bautista Ana Laura #15210793
 // 5 algoritmos de la secuencia de Fibonacci.
 
 //3.-En este cuarto trimestre se agregó una función que después de haber realizado las operaciones correspondientes, 
 //la función nos dará un resultado (retorno) este debe ser un valor entero (Int) Un ciclo (para) comienza donde k = 1,
 //comenzará a ciclar hasta que se convierta en (n), (n) representa el valor que se ingresará en la función 
 //dependiendo del ciclo (para) las variables (b, a) comenzarán a cambiar su resultado hasta el final del ciclo (para) 
 //el resultado se devolverá con (return)
 def  function3 ( n : Int ) :  Int  = {
 var  a  =  0 
 var  b  =  1 

 para (k < -  1 a n)
 {
   b = b + a
   a = b - a

 }
 
 return (a)
 }
  función3 ( 10 )