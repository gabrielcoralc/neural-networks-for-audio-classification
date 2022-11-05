# Neural networks for audio classification

Este proyecto fue desarrollado durante el primer semestre de la maestria de Matematicas aplicadas a ciencia de la computacion de la Universidad del Rosario.
El objetivo de este proyecto era poder aplicar a un caso practico el uso de redes neuronales, por lo que se se opto en desarrollar un modelo el cual pudiera identificar las vocales, dejando en claro que este modelo seria capaz de clasificar cualquier tipo de audio corto siempre y cuando se tenga una base de audios lo suficientemente grande para poder entrenar el modelo.

# Proceso para la creacion del modelo

## 1. Recolectar informacion para la base de datos

Uno de los principales retos que se tenia para este modelo es que no se puede encontrar facilmente una buena base de datos que tenga audios de las vocales en español por lo que para este caso fue necesario crear una base de datos de calidad que tuviera vocales de multiples personas.
Todos estos audios fueron recolectado con propositos academicos y son del libre uso. Los archivos se encuentran en este repositorio dentro de la carpeta **audios/** y se encuentran etiquetados de la siguiente manera: 

[Vocal del audio en mayuscula]_[identificador unico del audio]_[genero de la persona del audio]_[integrante que recolecto el audio].wav
A_001_mujer_paula.wav

## 2. Limpieza de audios

Los audios no son perfectos por lo que es necesario definir un proceso en donde se elimina el ruido del audio y se recorta el audio en el segmento donde se pronuncia la vocal,

## 3. Entrenamiento de modelos

Ya con nuestra informacion preprocesada y lista para ingresar a las redes neuronales, separamos nuestra base en set de entrenamiento y de pruebas, luego se empieza a probar diferentes estructuras de redes neuroanles y se elige la que mejor resultados presenta con el set de pruebas

## 4. Grabamos el mejor modelos y se crea un funcion para las predicciones
