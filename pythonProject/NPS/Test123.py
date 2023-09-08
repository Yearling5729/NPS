import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the data from the CSV file
data = pd.read_csv('NPS_13_Months.csv')
data.dropna(inplace=True)
# Convert the 'date_column' to a datetime object
data['DATE_OF_CALL'] = pd.to_datetime(data['DATE_OF_CALL'], format='%d/%m/%Y')
# Filter the DataFrame for June 2023
june_2023_data = data[(data['DATE_OF_CALL'].dt.year == 2023) & (data['DATE_OF_CALL'].dt.month == 6)]
print(june_2023_data)
