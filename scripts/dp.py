import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from arima_automated import fit_sarima_with_diagnostics
from ets_automated import ets_with_diagostics
from theta_automated import fit_theta

# --- Folder --- #
folder = '/Users/cemkarahan/Desktop/Python_Projects/stock_opt/'

# --- Read File --- #
df = pd.read_csv(folder + 'supply_chain_dataset1.csv')

# --- Field Definitions --- #
ts_columns = ['Date', 'Units_Sold']
sku = ['SKU_ID']
region = ['Region']
warehouse = ['Warehouse_ID']
stock_management = ['Date', 'Units_Sold', 'Inventory_Level', 'Supplier_Lead_Time_Days', 'Order_Quantity', 'Unit_Cost', 'Unit_Price']
promotion = ['Promotion_Flag']
moq = 10
penalty_cost = ['Unit']
horizon = 30

base_df = df[df['SKU_ID'] == 'SKU_1']
# --- TSA --- #
df_ts = base_df[promotion + ts_columns]
df_ts['Date'] = pd.to_datetime(df_ts['Date'], errors = 'coerce')
df_ts.set_index('Date', inplace = True)
df_ts.sort_index(inplace = True)
df_ts = df_ts.resample('D').agg({'Units_Sold': 'sum', 'Promotion_Flag': 'max'})

weeks_for_test = 4
test_size = weeks_for_test * 7
y = df_ts['Units_Sold']
train, test = y.iloc[:-test_size], y.iloc[-test_size:]
df_ts['Promotion_Flag_1'] = df_ts['Promotion_Flag'].shift(1).fillna(0)
df_ts['Promotion_Flag_2'] = df_ts['Promotion_Flag'].shift(2).fillna(0)
y_exog = df_ts[['Promotion_Flag']]
train_exog, test_exog = y_exog.iloc[:-test_size], y_exog[-test_size:]
model = ExponentialSmoothing(train, trend = 'add', seasonal = 'add', seasonal_periods = 7).fit(optimized = True)
forecast = model.forecast(steps = len(test))
ets_fitted = model.fittedvalues
residuals_ets = train - ets_fitted
reg_model = sm.OLS(residuals_ets, sm.add_constant(train_exog['Promotion_Flag'])).fit()
train_adjustment = reg_model.predict(sm.add_constant(train_exog['Promotion_Flag']))
test_adjustment = reg_model.predict(sm.add_constant(test_exog['Promotion_Flag']))
train_hybrid = ets_fitted + train_adjustment
forecast_hybrid = forecast + test_adjustment

residuals = test - forecast_hybrid
past_residuals = train - train_hybrid

bias = past_residuals.mean()
sigma = past_residuals.std(ddof = 1)


# --- Base DataFrame for DP --- #
base_df = df[promotion + stock_management]

# --- Formatting Data --- #
base_df['Date'] = pd.to_datetime(base_df['Date'], errors = 'coerce')
base_df.set_index('Date', inplace = True)
base_df.sort_index(inplace = True)
base_df = base_df.resample('D').agg({'Units_Sold': 'sum', 'Promotion_Flag': 'max'})
print(base_df)

# --- Forecast --- #


# --- Demand Distribution --- #
uncertainity = 
