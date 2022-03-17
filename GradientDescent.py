import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('student-mat.csv', sep=';')

df = df.loc[:, ["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Split data into inputs and target variable (X and y)
target = 'G3'
X = np.array(df.drop(columns = target, axis = 1))
y = np.array(df.loc[:, target])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)

m = x_train_scaled.shape[0]
x_b = np.c_[np.ones(m), x_train_scaled]
y_b = y_train.reshape(m, 1)

eta = 0.1
n_iterations = 10000

# random initialization of theta
theta = np.random.randn(6, 1)

for _ in range(n_iterations):
    gradients = 2/m * x_b.T.dot(x_b.dot(theta) - y_b)
    theta = theta - (eta * gradients)

x_new_b = np.c_[np.ones(x_test.shape[0]), x_test]
y_pred = x_new_b.dot(theta)

plt.style.use('ggplot')
plt.scatter(y_test, y_pred)
plt.show()
