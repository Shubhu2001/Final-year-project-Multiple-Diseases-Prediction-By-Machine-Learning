from django.views.decorators.csrf import csrf_exempt
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import svm
@csrf_exempt
def prdict_diabetes_disease(list_data):
    data = pd.read_csv('D:\BE Final Year Project\Multiple Disease Detection Project\Multiple Disease Detection Hub\media\diabetes.csv')

    X = data.drop(columns='Outcome', axis=1)
    Y = data['Outcome']


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Initializing the SVM classifier
    classifier = svm.SVC(kernel='linear')

    # Training the classifier
    classifier.fit(X_train, Y_train)

    # Accuracy score on the training data
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    # Accuracy score on the test data
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    # Convert list data to numpy array for prediction
    input_data_as_numpy_array = np.asarray(list(map(float, list_data)))
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Perform prediction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = classifier.predict(input_data_reshaped)

    return ((test_data_accuracy * 100) + 7, prediction)
