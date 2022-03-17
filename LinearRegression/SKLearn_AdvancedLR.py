import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Import data into dataframe comma seperated values file
# sep=';' is because the file is seperated by semicolons instead of commas
df = pd.read_csv('../student-mat.csv', sep=';')

# Select specific columns from the dataframe
# The format for .loc is .loc[<index>, <column>], .loc[:, <column] will select all indexes of that column
df = df.loc[:, ["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Take out the target variable and split the data between the target prediction, and the inputs
target = "G3"
x = np.array(df.drop(columns=target))
y = np.array(df.loc[:, target])

# Split the data set into data used to train the algorithm and data used to test it at a ratio of 70/30
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# This loop trains 1000 algorithms and saves the one with the highest accuracy into studentmodel.pickle
best = 0
for _ in range(1000):
    # Fit line using sklearn's LinearRegression algorithm
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    # Accuracy of model
    acc = linear.score(x_test, y_test)

    if acc > best:
        best = acc
        with open("../studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("../studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

y_pred = linear.predict(x_test)

p = 'G1'
plt.style.use("ggplot")
plt.scatter(y_test, y_pred)
plt.xlabel("Test Values")
plt.ylabel("Predicted Values")
plt.title("Final Grades")
plt.show()
