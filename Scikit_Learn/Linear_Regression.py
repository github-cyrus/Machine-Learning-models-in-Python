# Problem Statement 

''''Predict a student's final exam score based on the number of
hours they study'''

# Importing Package

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing data 

ashudata = {'SH':[2,3,4,5,6,7,8,9,10], 'ES':[50,60,75,80,85,90,95,98,99]}

# Putting data into DataFrame

df = pd.DataFrame(ashudata)

#print(df)

# Assigning X & Y Values 

x = df[['SH']]
y = df[['ES']] 

# Trainng Data on Model

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)

# Imporitng Model

model = LinearRegression()

# Fitting model in trainng set

model.fit(x_train,y_train)

# Taking user input

UI = int(input('Enter your study time :'))

# Predicting Values

predicted_score = model.predict([[UI]])

print("Predicated Score is :",predicted_score)