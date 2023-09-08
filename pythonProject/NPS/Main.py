import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the CSV file
# data = pd.read_csv('NPS_13_Months.csv')
# data.dropna(inplace=True)
# # print(data)
#
# # Calculate the correlation coefficient between NPS and ASA
# correlation_coefficient = data['NPS'].corr(data['ASA'])
#
# print("Correlation Coefficient between NPS and ASA:", correlation_coefficient)
#
# # Create a scatter plot with a regression line
# sns.lmplot(x='ASA', y='NPS', data=data)
#
# # Add labels and title
# plt.xlabel('Average Speed of Answer (ASA)')
# plt.ylabel('Net Promoter Score (NPS)')
# plt.title('Relationship between NPS and ASA')

# Show the plot
# plt.show()

# Generate some fake data
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm

import numpy as np
import pandas as pd
from statsmodels.formula.api import logit

# Create sample data
np.random.seed(12)
num_obs = 100
X1 = np.random.normal(2, 1, num_obs)
X2 = np.random.normal(1, 1, num_obs)
y = np.random.binomial(1, 0.5, num_obs)
#
# # Put data in pandas DataFrame
data = pd.DataFrame({"y": y, "X1": X1, "X2": X2})
#
# # Logistic regression model
model = logit("y ~ X1 + X2", data).fit()
#
# # Print model summary
# print(model.summary())

# Plot the data points
plt.scatter(data['X1'], data['y'], c=data['y'], s=100, cmap='coolwarm')

# Predict probability over a range of X1 values
X1_range = np.linspace(-2, 6, 100)
X2_fixed = 2.0
p = model.predict({"X1": X1_range, "X2": X2_fixed})

# Plot the probability curve
plt.plot(X1_range, p, color='k')

# Add threshold line at p=0.5
plt.hlines(0.5, -2, 6, colors='k', linestyles='dashed')

plt.show()