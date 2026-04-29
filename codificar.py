'''
Función de la Pregunta 1
'''

def desplazar(c, d):
   alfabeto = "abcdefghijklmnopqrstuvwxyz"
   i = 0
   while i<len(alfabeto) and alfabeto[i]!=c:
      i += 1
   if i<len(alfabeto):
      j = (i+d)%len(alfabeto)
      return alfabeto[j]
   return c

'''
Pregunta 2
Escriba el código de la función codificar
'''

def codificar(texto):
   # COMPLETE ACÁ


'''
Llamadas a la función codificar para verificar que está correcta
'''
print(codificar("edad#1"))
print(codificar("hola#3;mundo#2"))

'''
Pregunta 3
Descomente y complete el siguiente programa

continuar = True
while continuar:
   palabra = input("Ingrese una palabra: ")
   if palabra.lower()=="fin":
      continuar = False
   else:
      # COMPLETE ACÁ

'''
