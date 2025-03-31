# Problem Statement
'''A telecommunications company wants to reduce customer churn
by identifying customers at risk of leaving. They have historical
data on customer behavior and want to build a model to predict which
customers are most likely to churn.'''

# Importing Packages

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Importing Data

data = {'Age ': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
        'MonthlyCharge': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225],
        'chrun': [0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1]}

# Putting data into DataFrame

df = pd.DataFrame(data)
print(df)
# Assigning X & Y Values

X = df[['Age ', 'MonthlyCharge']]
y = df['chrun']

# Trainng Data on Model

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)

# Training Model

svc_model = SVC(kernel='linear')
svc_model.fit(x_train, y_train)

y_pred = svc_model.predict(x_test)

# Accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Confusion Matrix

#matrix = confusion_matrix(y_test, y_pred)
#print("Confusion Matrix:\n", matrix)

user_age = int(input("Enter the customer's age: "))
user_monthly_charge = int(input("Enter the customer's monthly charge: "))

new_data = pd.DataFrame({'Age ': [user_age], 'MonthlyCharge': [user_monthly_charge]})
prediction = svc_model.predict(new_data)
print("Prediction:", prediction[0])
if prediction[0] == 1:
    print("The customer is likely to churn.")
else:
    print("The customer is unlikely to churn.")