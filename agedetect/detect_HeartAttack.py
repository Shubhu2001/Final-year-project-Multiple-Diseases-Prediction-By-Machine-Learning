from django.views.decorators.csrf import csrf_exempt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
@csrf_exempt
def prdict_heart_disease(list_data):
   
        
        # Load data from the CSV file or another source
        # Replace this with your actual data loading logic
        data = pd.read_csv('D:\BE Final Year Project\Multiple Disease Detection Project\Multiple Disease Detection Hub\media\heart_disease_data.csv')
        #print(data.dtypes)

        # Handling missing values
        X = data.drop(columns='target', axis=1)
        Y = data['target']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

         # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
        
        model = LogisticRegression(max_iter=5000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X_scaled, Y, cv=5)

        # Perform predictions
        input_data_as_numpy_array = np.asarray(list_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = model.predict(input_data_reshaped)

        accuracy = (model.score(X_train, y_train)).mean() * 100
        
        return  cv_scores.mean()*100, prediction
