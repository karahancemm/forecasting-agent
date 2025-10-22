import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Theta
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# --- Eval Metrics --- #
def eval_metrics(y_true, y_pred, insample):
    mae = mean_absolute_error(y_true, y_pred)
    mse = root_mean_squared_error(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true.replace(0, np.nan))).dropna().mean()*100
    naive_mae = insample.diff().abs().dropna().mean() 
    mase = (np.abs(y_true - y_pred).mean()) / naive_mae
    return {'MAE': mae, 'RMSE': mse, 'MAPE%': mape, 'MASE': mase}


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
promotion = ['Promotion_Flag']

# --- Base Data Frame for TSA --- #
base_df = df[sku + ts_columns + promotion]
base_df = base_df[base_df['SKU_ID'] == 'SKU_1']
base_df = base_df[ts_columns + promotion]

# --- Formatting Data --- #
base_df['Date'] = pd.to_datetime(base_df['Date'], errors = 'coerce')
base_df.set_index('Date', inplace = True)
base_df.sort_index(inplace = True)
base_df = base_df.resample('D').agg({'Units_Sold': 'sum', 'Promotion_Flag': 'max'})
print(base_df)

# --- Train Test --- #
y_diff = base_df['Units_Sold'].diff().dropna()
weeks_for_test = 4
test_size = weeks_for_test * 7
y = base_df['Units_Sold']
train, test = y.iloc[:-test_size], y.iloc[-test_size:]

base_df['Promotion_Flag_1'] = base_df['Promotion_Flag'].shift(1).fillna(0)
base_df['Promotion_Flag_2'] = base_df['Promotion_Flag'].shift(2).fillna(0)
y_exog = base_df[['Promotion_Flag', 'Promotion_Flag_1', 'Promotion_Flag_2']]
y_exog = base_df[['Promotion_Flag']]

train_exog, test_exog = y_exog.iloc[:-test_size], y_exog[-test_size:]


model = StatsForecast(models = [Theta(season_length = 7)], freq = 'D', n_jobs = 1)
model.fit(train)
forecast = model.predict(h = len(test))

comparison_df = pd.DataFrame({'test': test.values, 'theta_forecast': forecast}, index = test.index)
print(comparison_df)