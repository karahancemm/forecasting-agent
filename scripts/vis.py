import pandas as pd
import io
#Import Primary Modules:
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import folium

folder = '/Users/cemkarahan/Desktop/Python_Projects/stock_opt/'

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv"

# Load directly into a DataFrame
df = pd.read_csv(url)
# Create bubble plot
plt.figure(figsize=(12,6))

plt.scatter(
    x=df["Year"],
    y=df["Month"],
    s=df["Automobile_Sales"] / 20,   # scale down size (adjust divisor if too large)
    c=df["Automobile_Sales"],
    alpha=0.6,
    cmap="viridis",
    edgecolors="w"
)

plt.title("Seasonality Impact on Automobile Sales (Bubble Plot)")
plt.xlabel("Year")
plt.ylabel("Month")
plt.colorbar(label="Automobile Sales")

plt.savefig("Bubble.png")
plt.show()