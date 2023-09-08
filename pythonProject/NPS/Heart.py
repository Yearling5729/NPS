import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm

heart= pd.read_csv('heart.data.csv')

heart = heart.rename(columns={'heart.disease':'heart_disease'})
# print(heart.head())

model_smoking = sm.ols('heart_disease ~ smoking', data=heart).fit()
print(model_smoking.summary())

print('\n=============\n')

# Create a dataframe to hold the date we want to make predictions for
data_range = pd.DataFrame({'smoking':np.arange(0, 50, 5)})

# Pass the data_range to make the predictions, but assign as a new column.
prediction_data = data_range.assign(prediction=model_smoking.predict(data_range))
print(prediction_data)