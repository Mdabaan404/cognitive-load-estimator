import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load data
data = pd.concat([
    pd.read_csv(f"data/low_load/{file}") for file in os.listdir("data/low_load") if file.endswith(".csv")
] + [
    pd.read_csv(f"data/high_load/{file}") for file in os.listdir("data/high_load") if file.endswith(".csv")
], ignore_index=True)

X = data.drop('label', axis=1)
y = data['label']

# Encode labels
y = y.map({'low_load': 0, 'high_load': 1})

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a better model
model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"Training Accuracy: {model.score(X_train, y_train)*100:.2f}%")
print(f"Test Accuracy: {model.score(X_test, y_test)*100:.2f}%")
