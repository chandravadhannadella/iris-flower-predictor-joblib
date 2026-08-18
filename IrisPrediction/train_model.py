# Step 1: Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 2: Load the Iris Flower dataset
iris = load_iris()

# Print dataset information
print("Feature names:", iris.feature_names)
print("Target class names:", iris.target_names)
print("Number of samples:", len(iris.data))

# Step 3: Feature and target separation
X = iris.data  # Four Iris measurements: sepal length, sepal width, petal length, petal width
y = iris.target  # Iris species labels

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n--- Train-Test Split ---")
print("Training set shape:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print("Testing set shape:")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

# Step 5: Create and train the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print("\n--- Model Training ---")
print("Decision Tree Classifier has been trained successfully!")
print(f"Model type: {type(model).__name__}")

# Step 6: Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Model serialization
joblib.dump(model, 'iris_model.pkl')

print("\n--- Model Serialization ---")
print("Model successfully saved as 'iris_model.pkl'")