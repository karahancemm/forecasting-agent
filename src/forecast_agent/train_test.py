import pandas as pd


def train_test_split(df, unique_id):
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
    unique_id = base_df['SKU_ID'].iat[0]
    base_df = base_df[ts_columns + promotion]

    # --- Formatting Data --- #
    base_df['Date'] = pd.to_datetime(base_df['Date'], errors = 'coerce')
    base_df.set_index('Date', inplace = True)
    base_df.sort_index(inplace = True)
    base_df = base_df.resample('D').agg({'Units_Sold': 'sum', 'Promotion_Flag': 'max'})
    print(base_df)

    # --- Train Test Split --- #
    y_diff = base_df['Units_Sold'].diff().dropna()
    weeks_for_test = 6
    test_size = weeks_for_test * 7
    y = base_df['Units_Sold']
    train, test = y.iloc[:-test_size], y.iloc[-test_size:]

    y_exog = base_df[['Promotion_Flag']]
    train_exog, test_exog = y_exog.iloc[:-test_size], y_exog[-test_size:]

    return train, test, train_exog, test_exog

# --- Folder --- #
folder = 'C:/Users/ckarahan/Desktop/Python Projects/Forecast-Agent/data/raw/'

# --- Read File --- #
df = pd.read_csv(folder + 'supply_chain_dataset1.csv')


print(df['SKU_ID'].iloc[0])
train, test, train_exog, test_exog = train_test_split(df, df['SKU_ID'].iloc[0])



