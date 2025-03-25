import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the training data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
train_data = pd.read_csv(train_url)

# Load the test data
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test_data = pd.read_csv(test_url)

print("Data Loaded Successfully!")

# Encode categorical variables
label_encoders = {}
for col in train_data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    
    # Check if column exists in test data
    if col in test_data.columns:
        # Apply transform but handle unseen labels by assigning them a default category (e.g., -1)
        test_data[col] = test_data[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    label_encoders[col] = le

if 'meal' in test_data.columns:
    test_data.drop(columns=['meal'], inplace=True)

# Separate features and target
y = train_data['meal']  # Target variable
X = train_data.drop(columns=['meal'])  # Features (excluding target)

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_data)

# Split into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model (XGBoost)
model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate model on validation set
y_pred_val = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_val)
print(f"Model Trained! Validation Accuracy: {accuracy:.4f}")
# Save the final trained model
joblib.dump(model, "model.pkl")

# Generate predictions on test data
pred = model.predict(X_test_scaled)

# Convert predictions to binary (0 or 1)
pred = np.where(pred > 0.5, 1, 0)

# Save predictions to CSV
np.savetxt("predictions.csv", pred, fmt="%d")

print("Predictions Saved! Model stored as model.pkl")