import pandas as pd
import numpy as np
import vix_utils as v
import asyncio as aio
import yfinance as yf
import logging
import sys
from itertools import chain

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def SpikeDate(date,data,threshold,window1,window2):
    
    vix_change = data.pct_change(periods=window1-1)
    # relative changes that are larger than the threshold
    test1 = vix_change > threshold  
    # the spike date must be the one that has the largest value during the past few days
    test2 = (data.rolling(window1).max() == data) 
    pot_spk = date[test1&test2]  # all potential spike dates 
    
    # commented because I don´t want that filter
    # date_change = (pot_spk.shift(-1) - pot_spk).dt.days # the days between this spike and next spike
    # test3 = date_change >= window2  # whether the days between two potential spikes are larger than window2
    # spike_date = pot_spk[test3]
    
    return pot_spk

async def do_load():
    async with aio.TaskGroup() as tg:
        t1 = tg.create_task(v.async_load_vix_term_structure())
        t2 = tg.create_task(v.async_get_vix_index_histories())

    return (t1.result(), t2.result())

symbol = "VIX"

vix_futures_history, vix_cash_history = aio.run(do_load())
vix_futures_history = v.load_vix_term_structure()

vix_cash_history = vix_cash_history[vix_cash_history.Symbol == symbol][["Trade Date", "Close"]]
vix_futures_history = vix_futures_history[["Trade Date", "Tenor_Days", "Tenor_Trade_Days", "Settle"]]
vix_futures_history['Num_Tenor'] = vix_futures_history.groupby('Trade Date')['Tenor_Days'].rank(method='first').astype(int)
# just the monthly
monthly = v.select_monthly_futures(vix_futures_history)

spx = yf.download("^GSPC", period="max", interval="1d")["Close"]

# Merge vix_cash_history, vix_futures_history, and spx
vix_cash_history['Trade Date'] = pd.to_datetime(vix_cash_history['Trade Date'])
vix_futures_history['Trade Date'] = pd.to_datetime(vix_futures_history['Trade Date'])
spx.index = pd.to_datetime(spx.index)

# Merge VIX cash with SPX
combined = pd.merge(vix_cash_history, spx.rename(columns={'Close': 'SP500 Close'}), left_on='Trade Date', right_index=True, how='left')

# Pivot VIX futures to get tenors as columns
vix_futures_pivot = vix_futures_history.pivot(index='Trade Date', columns='Num_Tenor', values='Settle')
vix_futures_pivot.columns = [f'Tenor {int(col)}' for col in vix_futures_pivot.columns]

# Merge the pivoted futures data
combined = pd.merge(combined, vix_futures_pivot, left_on='Trade Date', right_index=True, how='left')

# Filtrar columnas que empiezan con "Tenor "
tenor_columns = [col for col in combined.columns if col.startswith('Tenor ')]

# Eliminar filas donde todas las columnas Tenor X son NaN
combined = combined.dropna(subset=tenor_columns, how='all')

# Calculate week of the year
combined['Week of Year'] = combined['Trade Date'].dt.isocalendar().week

combined = combined.rename(columns={"Close": "VIX", "^GSPC": "SP500"})
# Calculate differences between tenors
tenors = ["VIX"] + [f'Tenor {i}' for i in range(1, 7)]
for i, tenor_i in enumerate(tenors):
    for j, tenor_j in enumerate(tenors):
        if j > i:
            combined[f'{tenor_i} - {tenor_j}'] = combined[tenor_i] - combined[tenor_j]

# Calcular cambios en los spreads
# Identificar columnas de spreads entre tenores y VIX-tenor spreads
spread_columns = [col for col in combined.columns if " - " in col and ("Tenor" in col or "VIX" in col)]

# Calcular las variaciones de los spreads en 1D, 2D, 3D y 4D
for days in [1, 2, 3, 4]:
    for col in spread_columns:
        combined[f'{col} Change {days}D'] = combined[col].diff(days)


# Calcular las variaciones de VIX y SP500 en porcentaje
def calculate_percentage_change(series):
    return series.pct_change() * 100

# Calcular variaciones porcentuales para VIX y SP500
combined['VIX Change 1D (%)'] = calculate_percentage_change(combined['VIX'])
combined['VIX Change 1W (%)'] = calculate_percentage_change(combined['VIX']).shift(5)  # 1 semana = 5 días
combined['VIX Change 1M (%)'] = calculate_percentage_change(combined['VIX']).shift(21)  # 1 mes = 21 días

combined['SP500 Change 1D (%)'] = calculate_percentage_change(combined['SP500'])
combined['SP500 Change 1W (%)'] = calculate_percentage_change(combined['SP500']).shift(5)  # 1 semana = 5 días
combined['SP500 Change 1M (%)'] = calculate_percentage_change(combined['SP500']).shift(21)  # 1 mes = 21 días

# Calculate slope of the futures curve (linear regression slope)
def calculate_slope(row):
    tenor_values = row[tenors].dropna().values
    tenor_values = tenor_values.astype(np.float64)
    if len(tenor_values) < 2:
        return np.nan
    x = np.arange(len(tenor_values))
    slope = np.polyfit(x, tenor_values, 1)[0]
    return slope

combined['Futures Curve Slope'] = combined.apply(calculate_slope, axis=1)

# Calculate skew of futures curve
combined['Futures Curve Skew'] = (combined['Tenor 5'] - combined['Tenor 1']) / (combined['Tenor 3'] - combined['Tenor 1'])

# Calculate VIX/SP500 ratio
combined['VIX/SP500 Ratio'] = combined['VIX'] / combined['SP500']

# Calculate RSI
def rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

combined['VIX RSI'] = rsi(combined['VIX'])

# Calculate MACD
def macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

combined['VIX MACD'], combined['VIX MACD Signal'] = macd(combined['VIX'])

# Calculate MACD
def macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

combined['VIX MACD'], combined['VIX MACD Signal'] = macd(combined['VIX'])

# Calcular la desviación estándar del VIX en los últimos 20 días
combined['VIX Std Dev 20D'] = combined['VIX'].rolling(window=20).std()

# Añadir el valor del VIX a futuro (5, 10 y 21 días laborables)
for days in [5, 10, 21]:
    combined[f'VIX in {days}D'] = combined['VIX'].shift(-days)

# Añadir el valor máximo del VIX en los siguientes 5, 10 y 21 días laborables
for days in [5, 10, 21]:
    combined[f'Max VIX in {days}D'] = combined['VIX'].rolling(window=days).max().shift(-days + 1)

#Añadir dia de Spikes
threshold = 0.2
window1 = 5
window2 = 5
spk_date = SpikeDate(combined["Trade Date"],combined.VIX,threshold,window1,window2)
combined["Spike"] = combined["Trade Date"].isin(spk_date).map({True: "Yes", False: "No"})

#Añadir spikes en 5 Días
combined["Spike in 5D"] = combined["Spike"].eq("Yes").rolling(window=5).max().shift(-5).map({1.0: "Yes", 0.0: "No"})

# Guardar el DataFrame final en un archivo Excel
combined.to_excel("output/vix_model_data.xlsx", index=False)
vix_cash_history.to_excel("output/vix_cash_history.xlsx", index=False)
vix_futures_history.to_excel("output/vix_futures_history.xlsx", index=False)
spx.to_excel("output/spx.xlsx", index=False)
