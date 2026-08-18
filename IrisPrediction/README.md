# Iris Flower Predictor

A simple machine learning project to predict Iris flower species based on sepal and petal measurements.

---

## 1. Project Title

**Iris Flower Species Prediction using Decision Tree Classifier**

---

## 2. Project Objective

The objective of this project is to build a machine learning model that can accurately classify Iris flowers into three species (Setosa, Versicolor, Virginica) based on four morphological measurements: sepal length, sepal width, petal length, and petal width.

---

## 3. Project Description

This project demonstrates a complete machine learning workflow for a classification problem. It uses the classic Iris dataset to train a Decision Tree classifier that learns to distinguish between three Iris species. The trained model is saved using Joblib for later use in making predictions on new, unseen data. The project is designed as a beginner-friendly AIML laboratory experiment with clear, step-by-step implementation.

---

## 4. Technologies Used

- **Python 3.7+** - Programming language
- **scikit-learn** - Machine learning library for model training and evaluation
- **joblib** - Model serialization and deserialization
- **NumPy** - Numerical computations (used in prediction script)

---

## 5. Dataset Used

**Iris Dataset** (built into scikit-learn)
- **Source**: `sklearn.datasets.load_iris()`
- **Samples**: 150 total (50 per species)
- **Features**: 4 numerical features
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Target Classes**: 3 species
  - Iris Setosa (class 0)
  - Iris Versicolor (class 1)
  - Iris Virginica (class 2)

---

## 6. Machine Learning Algorithm

**Decision Tree Classifier** (`sklearn.tree.DecisionTreeClassifier`)
- **Algorithm Type**: Supervised Learning - Classification
- **Working Principle**: Creates a tree-like model of decisions based on feature values
- **Parameters Used**: `random_state=42` for reproducibility
- **Why Decision Tree**: Simple to understand, interpretable, works well with small datasets, no feature scaling required

---

## 7. Model Serialization using Joblib

The trained model is saved to disk using **Joblib** (`joblib.dump()`):
- **File Name**: `iris_model.pkl`
- **Purpose**: Persist the trained model for later use without retraining
- **Loading**: `joblib.load('iris_model.pkl')` in the prediction script
- **Advantage**: Efficient for scikit-learn models containing large NumPy arrays

---

## 8. Project Structure

```
IrisPrediction/
├── train_model.py    # Train the ML model and save it
├── predict.py        # Load model and make predictions on user input
├── iris_model.pkl    # Saved trained model (generated after training)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## 9. Installation Steps

### Prerequisites
- Python 3.7 or higher installed on your system

### Install Dependencies
```bash
pip install scikit-learn joblib numpy
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

---

## 10. How to Train the Model

Run the training script:
```bash
python train_model.py
```

**What happens during training:**
1. Loads the Iris dataset from scikit-learn
2. Displays dataset information (feature names, target classes, sample count)
3. Splits data into training (80%) and testing (20%) sets
4. Creates and trains a Decision Tree Classifier
5. Evaluates model accuracy on the test set
6. Saves the trained model as `iris_model.pkl`

**Expected training output:**
```
Feature names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Target class names: ['setosa' 'versicolor' 'virginica']
Number of samples: 150

--- Train-Test Split ---
Training set shape:
  X_train: (120, 4)
  y_train: (120,)
Testing set shape:
  X_test: (30, 4)
  y_test: (30,)

--- Model Training ---
Decision Tree Classifier has been trained successfully!
Model type: DecisionTreeClassifier

--- Model Evaluation ---
Model Accuracy: 96.67%
```

---

## 11. How to Make Predictions

Run the prediction script:
```bash
python predict.py
```

**What happens during prediction:**
1. Loads the saved model from `iris_model.pkl`
2. Prompts user to enter four measurements:
   - Sepal Length (cm)
   - Sepal Width (cm)
   - Petal Length (cm)
   - Petal Width (cm)
3. Validates input (must be positive numbers)
4. Makes prediction using the loaded model
5. Displays the predicted Iris species

**Example interaction:**
```
--- Model Loading ---
Model successfully loaded from 'iris_model.pkl'
Model type: DecisionTreeClassifier

--- Iris Flower Prediction ---
Please enter the four iris flower measurements:
Sepal Length (cm): 5.1
Sepal Width (cm): 3.5
Petal Length (cm): 1.4
Petal Width (cm): 0.2

--- Prediction Result ---
Measurements: SL=5.1, SW=3.5, PL=1.4, PW=0.2
Predicted Iris Species: Setosa
```

---

## 12. Expected Output

### Training Phase
- Model accuracy typically **93-100%** on test data
- Model file `iris_model.pkl` created in project directory
- Console output showing dataset info, split sizes, and accuracy

### Prediction Phase
- Interactive input prompts for four measurements
- Input validation (rejects negative numbers and non-numeric input)
- Clear prediction result showing the species name (Setosa, Versicolor, or Virginica)

---

## 13. Conclusion

This project demonstrates a fundamental machine learning workflow suitable for AIML laboratory experiments:
1. **Data Loading** - Using built-in datasets
2. **Data Splitting** - Train-test split for evaluation
3. **Model Training** - Decision Tree classifier
4. **Model Evaluation** - Accuracy metric
5. **Model Persistence** - Joblib serialization
6. **Inference** - Real-time predictions on new data

The Decision Tree algorithm provides an interpretable model that achieves high accuracy on the Iris dataset. The modular design (separate training and prediction scripts) follows best practices for ML project structure. This project serves as an excellent foundation for understanding classification problems and can be extended by trying different algorithms (Random Forest, SVM, KNN) or adding features like cross-validation and hyperparameter tuning.

This will show example predictions for three different Iris species.

### 3. Use in Your Own Code

```python
from predict import predict_species

# Predict a single flower
species, probabilities = predict_species(5.1, 3.5, 1.4, 0.2)
print(f"Predicted species: {species}")
```

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 (setosa, versicolor, virginica)
- **Training Data**: 120 samples (80% of Iris dataset)
- **Test Data**: 30 samples (20% of Iris dataset)

## Dataset

The Iris dataset is a classic dataset in machine learning, containing 150 samples of Iris flowers with 4 features each, across 3 species.

## License

MIT License