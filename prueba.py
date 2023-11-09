import joblib
import numpy as np
from sklearn.preprocessing import normalize

# Cargar el modelo desde el archivo .pkl
loaded_model = joblib.load('modelo_entrenado.pkl')

# Datos de prueba (ajusta según tus necesidades)
datos_prueba = np.array([[13.64015675,15.46678066,23.94647217,34.34025192,44.2963028,74.73326111,12.07785988,36.0843811
]])

# Normalizar los datos de prueba
datos_prueba_normalizados = normalize(datos_prueba, axis=1, norm='max')

# Realizar predicciones con el modelo cargado
predicciones = loaded_model.predict(datos_prueba_normalizados)

# Imprimir las predicciones
#print("Datos de prueba original:")
#print(datos_prueba)
#print("Datos de prueba normalizados:")
#print(datos_prueba_normalizados)
print("Predicciones con datos de prueba normalizados:")
print(predicciones)
