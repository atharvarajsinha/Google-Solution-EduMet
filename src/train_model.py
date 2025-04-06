import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
from utils import PROCESSED_DATA_PATH, MODEL_PATH

# Loading processed data
df = pd.read_csv(PROCESSED_DATA_PATH)
X = df.drop(['School', 'Student_ID', 'Name', 'Final_Grade'], axis=1)
y = df['Final_Grade']

# Spliting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initializing and training Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)
joblib.dump(model, MODEL_PATH)

# Predicting and evaluating
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print("Model training complete!")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")
print("Model saved to:", MODEL_PATH)
