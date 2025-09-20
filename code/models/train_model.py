from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load data
data = load_wine()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model to the correct location
model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists
model_path = os.path.join(model_dir, 'wine_rf_model.joblib')
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")