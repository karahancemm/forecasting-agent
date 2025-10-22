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
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

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
print(base_df)


# --- Train Test --- #
y_diff = base_df['Units_Sold'].diff().dropna()
test_size = int(len(y_diff)*0.15)
y = base_df['Units_Sold']
train, test = y.iloc[:-test_size], y.iloc[-test_size:]

# --- Eval Metrics --- #
def eval_metrics(y_true, y_pred, insample):
    mae = mean_absolute_error(y_true, y_pred)
    mse = root_mean_squared_error(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true.replace(0, np.nan))).dropna().mean()*100
    naive_mae = insample.diff().abs().dropna().mean() 
    mase = (np.abs(y_true - y_pred).mean()) / naive_mae
    return {'MAE': mae, 'RMSE': mse, 'MAPE%': mape, 'MASE': mase}

# --- 7-14-30 Days Naive --- #
def seasonal_naive(train, test, season_len):
    pattern = train.iloc[season_len:]
    horizon = len(test)
    repeats = int(np.ceil(horizon/season_len))
    repeated = np.tile(pattern.values, repeats)[:horizon]
    return repeated

for i in [7, 14, 30]:
    naive_seasonal_pred = seasonal_naive(train, test, i)
    naive_scores = eval_metrics(test, naive_seasonal_pred, train)
    print(f'{i} score: {naive_scores}')
    comparison_df = pd.DataFrame({'actual': test.values, f'naive_pred_{i}': naive_seasonal_pred})
    #print(f'{i} df: \n', comparison_df)

naive_pred = pd.Series(train.iloc[-1], index = test.index)
naive_scores = eval_metrics(test, naive_pred, train)
print(f'1 scoew: {naive_scores}')
comparison_df = pd.DataFrame({'actual': test.values, f'naive_pred_{i}': naive_pred})


