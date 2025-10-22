import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Folder --- #
folder = '/Users/cemkarahan/Desktop/Python_Projects/stock_opt/'

# --- Read File --- #
df = pd.read_csv(folder + 'supply_chain_dataset1.csv')

# --- Field Definitions --- #
ts_columns = ['Date', 'Units_Sold']
sku = ['SKU_ID']
region = ['Region']
warehouse = ['Warehouse_ID']
stock_management = ['Units_Sold', 'Inventory_Level', 'Supplier_Lead_Time_Days', 'Reorder_Point', 'Order_Quantity', 'Unit_Cost', 'Unit_Price']

# --- Base Data Frame for TSA --- #
base_df = df[sku + ts_columns]
base_df = base_df[base_df['SKU_ID'] == 'SKU_1']
base_df = base_df[ts_columns]

# --- Formatting Data --- #
base_df['Date'] = pd.to_datetime(base_df['Date'], errors = 'coerce')
base_df.set_index('Date', inplace = True)
base_df.sort_index(inplace = True)
base_df = base_df.resample('D').sum()

# --- Plot Series --- #
"""sns.lineplot(x = base_df.index, y = 'Units_Sold', data = base_df)
plt.show()"""

# --- Checking yt = ybase + yseasonal + ytrend --- #
stl = STL(base_df['Units_Sold'], robust = True, period = 7).fit()

trend_strength = 1 - (stl.resid.var() / (stl.resid + stl.trend).var())
season_strength = 1 - (stl.resid.var() / (stl.resid + stl.seasonal).var())

print(f"Trend strength: {trend_strength:.2f}")
print(f"Seasonality strength: {season_strength:.2f}")

"""fig = stl.plot()
plt.show()
exit()
plt.plot(stl.trend)
plt.title('STL Trend Component')
plt.show()
print(stl.trend)
exit()"""

# --- Checking Stationarity --- #
y_diff = base_df['Units_Sold'].diff().dropna()
plot_acf(y_diff, lags = 60)
plt.show()


plot_pacf(y_diff, lags = 60, method = 'ywm')
plt.show()
exit()
# --- Augmented Dickey - Fuller --- #
adf = adfuller(y_diff)

# --- KPSS --- #
stat, p_value, lags, crit = kpss(y_diff, regression = 'c')

# --- Train Test --- #
test_size = int(len(y_diff)*0.2)
y = base_df['Units_Sold']
train, test = y.iloc[:-test_size], y.iloc[-test_size:]

arima_nodrift = ARIMA(train, order = [5, 1, 1], trend = 'n').fit()
arima_drift = ARIMA(train, order = [5, 1, 1], trend = 't').fit()

best = min([arima_nodrift, arima_drift], key = lambda r: (r.aic, r.bic))
print(best.summary)

