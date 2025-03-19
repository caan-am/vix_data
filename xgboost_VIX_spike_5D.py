import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos
model_data = pd.read_excel("output/model_data_spike_5d.xlsx")
model_data.set_index('Trade Date', inplace=True)
# model_data = model_data_date.drop(columns=["Trade Date"])

# Eliminar columnas innecesarias
if "Futures Curve Skew" in model_data.columns:
    model_data = model_data.drop(columns=["Futures Curve Skew"])

# Separar features (X) y target (y)
X = model_data.drop(columns=['Spike in 5D'])
y = model_data['Spike in 5D']

# Codificar la variable objetivo (Yes=1, No=0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir los datos en train (80%) y test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Calcular el balance de clases para ajustar `scale_pos_weight`
pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)  # (negatives / positives)

# Configurar el clasificador XGBoost para clasificación binaria
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=pos_weight,  # Manejo del desbalanceo
    use_label_encoder=False
)

# Hiperparámetros para el tuning
param_dist = {
    'n_estimators': [100, 250, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.5, 0.7, 1],
    'colsample_bytree': [0.5, 0.7, 1],
    'gamma': [0, 1, 5],
    'reg_lambda': [1, 10, 100]
}

# Validación cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# RandomizedSearchCV para búsqueda eficiente de hiperparámetros
random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=50,
    scoring='f1',  # Mejor métrica para desbalanceo
    cv=cv,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Ajustar el modelo con los datos de entrenamiento
random_search.fit(X_train, y_train)

# Resultados del mejor modelo
print("Mejores hiperparámetros:", random_search.best_params_)
print("Mejor puntuación de validación cruzada:", random_search.best_score_)

# Predicciones en el conjunto de prueba
best_model = random_search.best_estimator_
y_pred_encoded = best_model.predict(X_test)

# Decodificar las predicciones
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Evaluación del modelo con el set de test
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred_encoded))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_encoded))

print("\nPrecisión general:")
print(accuracy_score(y_test, y_pred_encoded))

# Predicciones del modelo: días que se predice un spike
spike_dates = model_data.index[y_pred_encoded == 1]

# Crear el gráfico
plt.figure(figsize=(10, 6))

# Graficar el VIX
plt.plot(model_data.index, model_data['VIX'], label='VIX', color='blue')

# Graficar los puntos verdes donde el modelo predice un spike
plt.scatter(spike_dates, model_data.loc[spike_dates, 'VIX'], color='green', label='Predicción Spike', marker='o')

# Personalización del gráfico
plt.title('Predicción de Spikes en el VIX')
plt.xlabel('Fecha')
plt.ylabel('Valor del VIX')
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()