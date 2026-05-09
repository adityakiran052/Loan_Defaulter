import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

# 1. Create the directory
os.makedirs('models', exist_ok=True)

# 2. MATCH THE API: Your error says the API is sending 24 features
num_features = 24 

# 3. Train the dummy model with 24 features
mock_model = LogisticRegression()
X_fake = np.zeros((2, num_features)) 
y_fake = np.array([0, 1])
mock_model.fit(X_fake, y_fake)

# 4. Save it
joblib.dump(mock_model, "models/model.pkl")
print(f"✅ Dummy model RE-SYNCED to {num_features} features.")