
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load sample data (Iris dataset)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Prepare the results in a DataFrame
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results['Accuracy'] = accuracy

# Print the results
print(results)

# Save the model and results to a file
results.to_csv('simple_ml_project_results.csv', index=False)
