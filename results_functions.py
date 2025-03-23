from sklearn.metrics import  confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_spikes_predicted(y_pred,y,spikes_filtered):
    
    y_pred_series = pd.Series(y_pred, index=y.index)
    df = pd.DataFrame({"Spike real en 5 D": y, "Spike predicted?": y_pred_series})
    df["Spike"] = ["Yes" if date in spikes_filtered.index else "No" for date in df.index]

    # Convertir "Yes" y "No" a 1 y 0 para las columnas necesarias
    df["Spike predicted?"] = df["Spike predicted?"].map({"Yes": 1, "No": 0})
    df["Spike"] = df["Spike"].map({"Yes": 1, "No": 0})

    # Inicializamos la columna "Spike predicted actually" como "No"
    df["Spike predicted actually"] = "No"

    #Copia de spike predicted, donde sustituyo los yeses de un spike por un yes y el resto se quedan
    spike_predicted_modified = df["Spike predicted?"].copy() 
    # Iteramos sobre las fechas donde "Spike" es 1 (es decir, hay un spike)
    for idx in df[df["Spike"] == 1].index:
        # Obtener los 5 días previos al índice actual
        prev_dates = df.index[df.index.get_loc(idx) - 5 : df.index.get_loc(idx)]  # Conseguimos los 5 días previos
        # Seleccionamos las fechas de "Spike predicted?" para esos 5 días previos
        last_5_days = df.loc[prev_dates, "Spike predicted?"]

        # Si alguno de esos días tiene "Yes" (1), marcamos "Yes" en la columna "Spike predicted actually"
        if last_5_days.sum() > 0:  # Si hay al menos un "Yes" en los 5 días previos
            df.at[idx, "Spike predicted actually"] = "Yes"
            spike_predicted_modified.at[idx] = 1
            for date in prev_dates:
                if date not in df[df["Spike"] == 1].index:
                    spike_predicted_modified.at[date] = 0

    df["Spike predicted confusion matrix"] = spike_predicted_modified
    # Convertimos de vuelta a "Yes" y "No"
    df["Spike predicted?"] = df["Spike predicted?"].map({1: "Yes", 0: "No"})
    df["Spike predicted confusion matrix"] = df["Spike predicted confusion matrix"].map({1: "Yes", 0: "No"})
    # Volvemos a convertir "Spike predicted?" a "Yes"/"No"
    df["Spike"] = df["Spike"].map({1: "Yes", 0: "No"})
    # Mostrar el DataFrame resultante
    return df

def plot_confusion_matrix(y_true, y_pred, labels=None, cmap='Blues'):
    """
    Genera y muestra una matriz de confusión con un estilo atractivo.

    Parámetros:
    - y_true: array-like. Valores reales de las clases.
    - y_pred: array-like. Valores predichos por el modelo.
    - labels: list, opcional. Etiquetas de las clases.
    - cmap: str, opcional. Mapa de colores para la visualización.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = np.unique(y_true)  # Genera etiquetas automáticamente si no se proporcionan

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.show()


def draw_predictions(y_pred_encoded, all_data, filtered_spikes):
    # Identificar los días donde el modelo predice un spike (predicción = 1)
    spike_dates = all_data.index[y_pred_encoded == 1]

    # Crear el gráfico
    plt.figure(figsize=(10, 6))

    # Graficar el VIX
    plt.plot(all_data.index, all_data['VIX'], label='VIX', color='blue')

    # Marcar los puntos verdes donde el modelo predice un spike
    plt.scatter(spike_dates, all_data.loc[spike_dates, 'VIX'], color='green', label='Predicción Spike', marker='o')

    
    # Marcar los puntos rojos donde hay un spike
    plt.scatter(filtered_spikes.index,filtered_spikes["VIX"], color='red', label='Spike', marker='x', alpha=0.5, s=100)

    # Personalización del gráfico
    plt.title('Predicción de Spikes en el VIX')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del VIX')
    plt.legend()
    plt.grid(True)

    # Mostrar el gráfico
    plt.show()