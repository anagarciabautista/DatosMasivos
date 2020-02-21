// Práctica 2
//Garcia Bautista Ana Laura #15210793


// 1. Crea una lista llamad "lista" con los elementos "rojo", "blanco", "negro"
import scala.collection.mutable.ListBuffer
var lista = collection.mutable.ListBuffer("rojo","blanco","negro")

// 2. Añadir 5 elementos mas a "lista" "verde", "amarillo", "azul", "naranja", "perla"
lista += "verde"
lista += "amarillo"
lista += "azul"
lista += "naranja"
lista += "perla"

// 3. Traer los elementos de "lista" "verde", "amarillo", "azul"
lista slice (3,6)

// 4. Crea un arreglo de número en rango del 1-1000 en pasos de 5 en 5
Array.range(1, 1000, 5)

// 5. Cuales son los elementos unicos de la lista Lista (1,3,3,4,6,7,3,7) usan conversion a conjuntos
lista.toSet

// 6. Crea un mapa mutable llamado nombres que contenga los siguientes "José", 20, "Luis", 24, "Ana", 23, "Susana", "27"
var names = collection.mutable.Mapa(("Jose", 20), ("Luis", 24), ("Ana", 23) ("Susana", 27))

// 6 a. Imprime todas las llaves del mapa
nombres.claves

// 7 b. Agrega el siguiente valor al mapa ("Miguel", 23)
nombres += ("Miguel" -> 23)