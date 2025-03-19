import pandas as pd
import numpy as np

model_data = pd.read_excel("output/vix_model_data.xlsx")

model_data_needed_cols = model_data[['VIX', 'SP500', 'Tenor 1', 'Tenor 2', 'Tenor 3', 'Tenor 4', 'Week of Year', 'VIX - Tenor 1', 'VIX - Tenor 2', 'VIX - Tenor 3', 'VIX - Tenor 4', 
                                    'Tenor 1 - Tenor 2', 'Tenor 1 - Tenor 3', 'Tenor 1 - Tenor 4', 'Tenor 2 - Tenor 3', 'Tenor 2 - Tenor 4','Tenor 3 - Tenor 4', 
                                    'VIX - Tenor 1 Change 1D', 'VIX - Tenor 2 Change 1D', 'VIX - Tenor 3 Change 1D', 'VIX - Tenor 4 Change 1D', 'Tenor 1 - Tenor 2 Change 1D', 
                                    'Tenor 1 - Tenor 3 Change 1D', 'Tenor 1 - Tenor 4 Change 1D', 'Tenor 2 - Tenor 3 Change 1D', 'Tenor 2 - Tenor 4 Change 1D', 'Tenor 3 - Tenor 4 Change 1D',
                                    'VIX - Tenor 1 Change 2D', 'VIX - Tenor 2 Change 2D', 'VIX - Tenor 3 Change 2D', 'VIX - Tenor 4 Change 2D','Tenor 1 - Tenor 2 Change 2D', 
                                    'Tenor 1 - Tenor 3 Change 2D', 'Tenor 1 - Tenor 4 Change 2D', 'Tenor 2 - Tenor 3 Change 2D', 'Tenor 2 - Tenor 4 Change 2D', 'Tenor 3 - Tenor 4 Change 2D',
                                    'VIX - Tenor 1 Change 3D', 'VIX - Tenor 2 Change 3D', 'VIX - Tenor 3 Change 3D', 'VIX - Tenor 4 Change 3D', 'Tenor 1 - Tenor 2 Change 3D',
                                    'Tenor 1 - Tenor 3 Change 3D', 'Tenor 1 - Tenor 4 Change 3D', 'Tenor 2 - Tenor 3 Change 3D', 'Tenor 2 - Tenor 4 Change 3D', 'Tenor 3 - Tenor 4 Change 3D',
                                    'VIX - Tenor 1 Change 4D', 'VIX - Tenor 2 Change 4D', 'VIX - Tenor 3 Change 4D', 'VIX - Tenor 4 Change 4D', 'Tenor 1 - Tenor 2 Change 4D',
                                    'Tenor 1 - Tenor 3 Change 4D', 'Tenor 1 - Tenor 4 Change 4D', 'Tenor 2 - Tenor 3 Change 4D', 'Tenor 2 - Tenor 4 Change 4D','Tenor 3 - Tenor 4 Change 4D', 
                                    'VIX Change 1D (%)', 'VIX Change 1W (%)', 'VIX Change 1M (%)', 'SP500 Change 1D (%)', 'SP500 Change 1W (%)', 'SP500 Change 1M (%)', 'Futures Curve Slope', 
                                    'Futures Curve Skew', 'VIX/SP500 Ratio', 'VIX RSI', 'VIX MACD', 'VIX MACD Signal', 'VIX in 5D', 'VIX in 10D', 'VIX in 21D', 'Max VIX in 5D', 'Max VIX in 10D',
                                    'Max VIX in 21D']]

model_data_needed_cols = model_data_needed_cols.dropna(how="any")

bins = [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, float('inf')]
labels = [
    "0-10", "10-11", "11-12", "12-13", "13-14", "14-15", "15-16", "16-17", "17-18", "18-19", "19-20", 
    "20-21", "21-22", "22-23", "23-24", "24-25", "25-26", "26-27", "27-28", "28-29", "29-30", "30-35", 
    "35-40", "40-45", "45-50", "50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80+"
]

predict_cols = ['VIX in 5D', 'VIX in 10D', 'VIX in 21D', 'Max VIX in 5D', 'Max VIX in 10D', 'Max VIX in 21D']

for predict_col in predict_cols:
    # Crear la nueva columna con los rangos
    model_data_needed_cols["Rango "+predict_col] = pd.cut(model_data_needed_cols[predict_col], bins=bins, labels=labels, right=False)

model_data_needed_cols = model_data_needed_cols.drop(columns=predict_cols)

predict_cols = ['Rango VIX in 5D', 'Rango VIX in 10D', 'Rango VIX in 21D', 'Rango Max VIX in 5D', 'Rango Max VIX in 10D', 'Rango Max VIX in 21D']


def eliminar_elemento(lista, elemento_a_eliminar):
    nueva_lista = [x for x in lista if x != elemento_a_eliminar]
    return nueva_lista

dict_df_models = {}
for predict_col in predict_cols:
    df = model_data_needed_cols.drop(columns=eliminar_elemento(predict_cols, predict_col))
    dict_df_models[predict_col] =  df



# Guardar cada DataFrame como un archivo Excel
for nombre_archivo, df in dict_df_models.items():
    df.to_excel("output/model_data_" + nombre_archivo  + ".xlsx", index=False)