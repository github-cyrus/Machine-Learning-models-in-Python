import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Simulate the dataset
np.random.seed(42)
n_samples = 1000

# Generate random data for features
age = np.random.randint(18, 60, size=n_samples)  # Age between 18 and 60
time_spent = np.random.uniform(1, 20, size=n_samples)  # Time spent on website (1 to 20 minutes)
added_to_cart = np.random.randint(0, 2, size=n_samples)  # 0 or 1 (No or Yes)

# Generate target variable based on some conditions
purchase = ((age > 30) & (time_spent > 10) & (added_to_cart == 1)).astype(int)

# Create a DataFrame
data = pd.DataFrame({
    'Age': age,
    'Time_Spent': time_spent,
    'Added_to_Cart': added_to_cart,
    'Purchase': purchase
})

# 2. Split the data into training and testing sets
X = data[['Age', 'Time_Spent', 'Added_to_Cart']]  # Features
y = data['Purchase']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Function to predict purchase for a new customer
def predict_purchase(age, time_spent, added_to_cart):
    # Prepare input data as a 2D array
    customer_data = np.array([[age, time_spent, added_to_cart]])
    prediction = model.predict(customer_data)  # Predict using trained model
    
    # Output
    if prediction[0] == 1:
        return "The customer is likely to purchase a high-value product."
    else:
        return "The customer is unlikely to purchase a high-value product."

# 5. Test with a new customer's data
customer_age = int(input("Enter the customer's age: "))        # Age of the customer
customer_time_spent = int(input("Enter the customer's time spent on the website: ")) # Time spent on the website
customer_added_to_cart = int(input("Did the customer add items to the cart? (1 = Yes, 0 = No): ")) # 1 # Whether the customer added items to the cart (1 = Yes, 0 = No)

# Predict and print the result
result = predict_purchase(customer_age, customer_time_spent, customer_added_to_cart)
print(result)
