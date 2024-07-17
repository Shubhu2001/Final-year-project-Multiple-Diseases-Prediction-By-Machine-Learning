import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
heart_data = pd.read_csv('../media/heart_disease_data.csv')

heart_data.head()

heart_data.info()

heart_data.describe()

heart_data['target'].value_counts()

# Handling the missing values
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2, )

print(X.shape, X_train.shape, X_test.shape)

# Use GradientBoostingClassifier
nn_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
nn_model.fit(X_train, Y_train)

# Accuracy on Training Data
X_train_prediction = nn_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training Data: ', training_data_accuracy*100)

# Accuracy on Test Data
X_test_prediction = nn_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test Data', test_data_accuracy * 100)

# input_data = (41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2)
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = nn_model.predict(input_data_reshaped)
print(prediction)

if prediction == 1:
    print("You have a chance of Heart Attack")
else:
    print("You're healthy")
