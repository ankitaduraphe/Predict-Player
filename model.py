# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("predictive_batting_1_2_2019_final1.csv")

X = dataset.iloc[:, :9]
y = dataset.iloc[:, -1]

# Splitting Train and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# SVM model
from sklearn import svm

regressor = svm.SVC(kernel='linear')

# Fitting model with training data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1, 1, 0, 0, 1, 0, 6, 3, 0]]))