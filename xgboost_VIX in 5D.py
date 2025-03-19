import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# Cargar los datos
model_data = pd.read_excel("output/model_data_Rango VIX in 5D.xlsx")
model_data = model_data.drop(columns=["Futures Curve Skew"])
# Obtener las clases únicas (los rangos)
class_labels = model_data['Rango VIX in 5D'].unique()

# Crear un mapeo de etiquetas a enteros
class_mapping = {label: idx for idx, label in enumerate(class_labels)}

# Aplicar el mapeo para convertir las clases a enteros
model_data['Rango VIX in 5D Encoded'] = model_data['Rango VIX in 5D'].map(class_mapping)

# Separar features (X) y target (y)
X = model_data.drop(columns=['Rango VIX in 5D', 'Rango VIX in 5D Encoded'])
y = model_data['Rango VIX in 5D Encoded']

print("Class mapping:", class_mapping)

# Dividir los datos en train (80%) y test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Configurar el clasificador XGBoost para multiclase
xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=len(class_mapping), eval_metric='mlogloss')

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
    scoring='accuracy',
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
inv_class_mapping = {v: k for k, v in class_mapping.items()}
y_pred = pd.Series(y_pred_encoded).map(inv_class_mapping)

print("Predicciones decodificadas:", y_pred.head())

# Evaluación del modelo con el set de test
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred_encoded))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_encoded))

print("\nPrecisión general:")
print(accuracy_score(y_test, y_pred_encoded))
