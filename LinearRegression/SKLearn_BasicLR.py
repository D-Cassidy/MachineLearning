import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns

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
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3, random_state=42)

# Fit line using sklearn's LinearRegression algorithm
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

# Make predictions on x_test
y_pred = linear.predict(x_test)

# Print out the prediction on each student along with the inputs, and the actual final grade
for i in range(len(y_pred)):
    print(y_pred[i], x_test[i], y_test[i])

# Scores and shows that accuracy of the model based on the test set
acc = linear.score(x_test, y_test)
print("Accuracy:", acc)

# Graphs the actual final grades of the test values versus the predicted final grades
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
