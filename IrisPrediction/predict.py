# Step 1: Import required libraries
import joblib
import numpy as np

# Step 2: Load the pre-trained model
model = joblib.load('iris_model.pkl')

print("--- Model Loading ---")
print("Model successfully loaded from 'iris_model.pkl'")
print(f"Model type: {type(model).__name__}")

# Step 3: Define species names
species_names = ['Setosa', 'Versicolor', 'Virginica']

# Step 4: Get user input for iris measurements
print("\n--- Iris Flower Prediction ---")
print("Please enter the four iris flower measurements:")

def get_float_input(prompt):
    """Helper function to get float input from user with error handling."""
    while True:
        try:
            value = float(input(prompt))
            if value < 0:
                print("Error: Measurements must be positive. Please try again.")
                continue
            return value
        except ValueError:
            print("Error: Please enter a valid number.")

sepal_length = get_float_input("Sepal Length (cm): ")
sepal_width = get_float_input("Sepal Width (cm): ")
petal_length = get_float_input("Petal Length (cm): ")
petal_width = get_float_input("Petal Width (cm): ")

# Step 5: Create sample array in correct 2D format
sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Step 6: Make a prediction
predicted_class = model.predict(sample)[0]
predicted_species = species_names[predicted_class]

# Step 7: Display the result
print("\n--- Prediction Result ---")
print(f"Measurements: SL={sepal_length}, SW={sepal_width}, PL={petal_length}, PW={petal_width}")
print(f"Predicted Iris Species: {predicted_species}")

