import joblib
import pandas as pd
import numpy as np  # Asegúrate de agregar esta línea
from sklearn.preprocessing import normalize

# Cargar el modelo desde el archivo .pkl
loaded_model = joblib.load('modelo_entrenado.pkl')

# Cargar datos desde el archivo Excel
archivo_excel = 'datossensor2_ordenados.xlsx'  # Reemplaza con el nombre de tu archivo Excel
datos = pd.read_excel(archivo_excel, header=None)

# Iterar sobre cada fila, normalizar y hacer predicciones
for indice, fila in datos.iterrows():
    # Convertir la fila a un array de numpy
    datos_fila = np.array([fila])

    # Normalizar los datos de la fila
    datos_fila_normalizados = normalize(datos_fila, axis=1, norm='max')

    # Realizar predicciones con el modelo cargado
    predicciones = loaded_model.predict(datos_fila_normalizados)

    # Imprimir resultados
    print(f"Predicciones para la fila {indice + 1}: {predicciones}")
