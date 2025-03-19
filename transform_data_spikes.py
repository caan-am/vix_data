import pandas as pd
import numpy as np

model_data = pd.read_excel("output/vix_model_data.xlsx")

model_data_needed_cols = model_data[['Trade Date','VIX', 'SP500', 'Tenor 1', 'Tenor 2', 'Tenor 3', 'Tenor 4', 'Week of Year', 'VIX - Tenor 1', 'VIX - Tenor 2', 'VIX - Tenor 3', 'VIX - Tenor 4', 
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
                                    'Futures Curve Skew', 'VIX/SP500 Ratio', 'VIX RSI', 'VIX MACD', 'VIX MACD Signal', 'VIX Std Dev 20D', "Spike in 5D"]]

model_data_spike_5d = model_data_needed_cols.dropna(how="any")


model_data_spike_5d.to_excel("output/model_data_spike_5d.xlsx", index=False)