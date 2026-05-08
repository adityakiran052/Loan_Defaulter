import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

# 1. Create the directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# 2. Train a "stub" model
# Your build_feature_vector creates 25 features (7 base + 2 engineered + 16 categorical)
# We need to match that number exactly so the model doesn't crash.
mock_model = LogisticRegression()
X_fake = np.zeros((2, 25)) 
y_fake = np.array([0, 1])
mock_model.fit(X_fake, y_fake)

# 3. Save it exactly where your api.py expects it
joblib.dump(mock_model, "models/model.pkl")
print("✅ Dummy model saved to models/model.pkl")