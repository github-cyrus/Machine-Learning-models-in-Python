import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Define the data and labels correctly
data = np.array([
    [25, 50000, 2],
    [30, 60000, 3],
    [35, 70000, 4],
    [40, 80000, 5],
    [45, 90000, 6],
    [50, 100000, 7],
    [55, 110000, 8],
    [60, 120000, 9],
    [65, 130000, 10],
    [70, 140000, 11],
    [75, 150000, 12]
])

labels = np.array([1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define the KNN model with 3 neighbors and Euclidean distance metric
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(x_train, y_train)

# Define the decision tree model
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

# Define the random forest model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Evaluate the models using accuracy score and confusion matrix
knn_accuracy = knn.score(x_test, y_test)
dt_accuracy = dt.score(x_test, y_test)
rf_accuracy = rf.score(x_test, y_test)

print("KNN Accuracy:", knn_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, knn.predict(x_test)))
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt.predict(x_test)))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf.predict(x_test)))
# Make predictions on new data
# new_data = np.array([[28, 75000, 3]])
# prediction = knn.predict(new_data)
# print("Prediction:", prediction)

# Make predictions on new data
user_age = int(input("Enter the customer's age: "))
user_monthly_charge = int(input("Enter the customer's monthly charge: "))
user_time_spent = int(input("Enter the customer's time spent: "))

new_data = np.array([[user_age, user_monthly_charge, user_time_spent]])
prediction = knn.predict(new_data)
print("Prediction:", prediction[0])