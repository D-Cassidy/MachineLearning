import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns

plt.style.use('ggplot')

# Import data to dataframe df
df = pd.read_csv('student-mat.csv', sep=';')
print(df.shape)
with pd.option_context('display.max_columns', 40):
    print(df.describe(include='all'))

# numerical columns
numerical_columns = df.select_dtypes(include = [np.number])
print("Numerical Columns:\n", numerical_columns.columns)

# categorical columns
categorical_columns = df.select_dtypes(include = [np.object])
print("Categorical Columns: \n", categorical_columns.columns)

# Missing value percentages in each row
"""percentage = df.isnull().mean() * 100
percentage = percentage.to_frame("nulls")
percentage.sort_values("nulls", inplace = True, ascending = False)
for index, row in percentage.iterrows():
    print(index, row[0])"""
# There are no missing values :)

# Find any correlation between data points
correlation = numerical_columns.corr()
sns.heatmap(correlation, annot=True)
plt.show()
