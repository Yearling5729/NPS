import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('NPS_13_Months.csv')
data.dropna(inplace=True)
# print(data.describe())

# Plot the time series
plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
plt.plot(data['DATE_OF_CALL'], data['NPS'], marker='o', linestyle='-')
plt.title("NPS Time Series")
plt.xlabel("Date")
plt.ylabel("NPS")
plt.grid(True)
plt.show()