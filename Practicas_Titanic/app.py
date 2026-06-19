# Importa la clase Flask para el servidor web, 'request' para capturar los datos entrantes y 'jsonify' para estructurar la respuesta en JSON
from flask import Flask, request, jsonify

# Importa pickle para cargar/deserializar el pipeline que guardamos desde el notebook de entrenamiento
import pickle

# Importa pandas ya que el pipeline necesita recibir un DataFrame para mantener los nombres de las columnas originales
import pandas as pd

# Inicializa la aplicación Flask para exponer nuestros endpoints HTTP
app = Flask(__name__)

# Abre el archivo 'pipeline.pkl' en modo lectura de bytes ('rb') para cargar el modelo final en la memoria del servidor
with open('pipeline.pkl', 'rb') as archivo_modelo:
    # Carga el pipeline completo (que incluye el ColumnTransformer de ingeniería de características y el algoritmo ganador)
    modelo = pickle.load(archivo_modelo)

# Define la ruta '/predecir' encargada de recibir las peticiones de los clientes mediante el método HTTP POST
@app.route('/predecir', methods=['POST'])
def predecir(): # Función que se ejecuta cada vez que un cliente envía datos al endpoint
    
    # Captura el cuerpo de la petición HTTP y lo convierte en un diccionario de Python de forma automática
    data = request.get_json()

    # Transforma el diccionario en un DataFrame de Pandas de una sola fila (encapsulando el JSON entre corchetes '[data]')
    input_data = pd.DataFrame([data])

    # Envía el DataFrame al pipeline; este aplica de forma transparente los Encoders, Transformers y genera la predicción final
    prediccion = modelo.predict(input_data)
    
    # Mapea el resultado numérico de la predicción (0 o 1) a una clave de negocio ('Survived') convirtiéndolo a entero nativo de Python
    output = {'Survived': int(prediccion[0])}
    
    # Convierte el diccionario de respuesta en formato JSON oficial y lo envía de vuelta al cliente con un código de estado HTTP 200 (OK)
    return jsonify({'Survived': int(prediccion[0])})

# Comprueba si el script se está ejecutando directamente desde la consola del sistema operativo
if __name__ == '__main__':
    # Lanza el servidor local en modo de desarrollo (debug=True), lo que permite reiniciar el servidor si modificas el código
        app.run(debug=True)