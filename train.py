import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib

# Cargar datos desde el archivo excel
pet = pd.read_excel('datossensor2.xlsx').values

# Transponer la matriz
pet = pet.T

# Obtener datos para el objetivo
objetivo = pet[8, :]

# Normalizar los datos
pet[0:8, :] = normalize(pet[0:8, :], axis=1, norm='max')

# Dividir datos para entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(pet[0:8, :].T, objetivo, test_size=0.3, random_state=42)

# Convertir y_train a enteros
y_train = y_train.astype(int)

# Aplicar SMOTE para realizar sobremuestreo de la clase minoritaria
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Crear y entrenar la red neuronal
clf = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', max_iter=3000,
                    learning_rate_init=0.005, batch_size=128, alpha=0.0001, random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Realizar predicciones en datos de prueba
predictions = clf.predict(X_test)

# Calcular eficiencia
accuracy = accuracy_score(y_test, predictions)
print("Eficiencia:", accuracy)

# Crear y visualizar la matriz de confusión
conf_matrix = confusion_matrix(y_test, predictions)
print("Matriz de Confusión:")
print(conf_matrix)

# Visualizar la matriz de confusión
plt.figure()
plt.matshow(conf_matrix, cmap='Blues')
plt.title('Matriz de Confusión')
plt.colorbar()
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.show()
joblib.dump(clf, 'modelo_entrenado.pkl')
