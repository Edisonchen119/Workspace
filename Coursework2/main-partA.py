import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load and preprocess the training data
train_data = pd.read_csv('TrainingDataBinary.csv')

# Separate features and labels
features = train_data.iloc[:, :128]
labels = train_data.iloc[:, -1]

# Step 2: Select a machine learning model (Random Forest Classifier)
model = RandomForestClassifier()

# Step 3: Train the model
# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.3, random_state=45)

# Fit the model to the training data
model.fit(X_train, y_train)

# Evaluate the model's performance on the validation set
val_predictions = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", val_accuracy)

# Step 4: Load and preprocess the testing data
test_data = pd.read_csv('TestingDataBinary.csv')

# Separate features from the testing data
test_features = test_data.values  # Use .values to access the underlying NumPy array

# Step 5: Predict labels for the testing data
test_predictions = model.predict(test_features)

# Step 6: Generate the TestingResultsBinary.csv file
test_results = pd.DataFrame({'Label': test_predictions})
test_results.to_csv('TestingResultsBinary.csv', index=False)

