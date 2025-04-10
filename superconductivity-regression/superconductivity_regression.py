
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Fetch dataset
superconductivty_data = fetch_ucirepo(id=464) 
  
# Data (as pandas dataframes)
X = superconductivty_data.data.features 
y = superconductivty_data.data.targets 
  
# Metadata
print(superconductivty_data.metadata) 
  
# Variable information
print(superconductivty_data.variables) 

# Display shapes
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Show the data
print("\nFeatures:")
print(X.head())

print("\nTarget (critical temperature):")
print(y.head())

plt.hist(y, bins=50, color="skyblue")
plt.title("Distribution of Critical Temperatures")
plt.xlabel("Critical Temperature (K)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Now we have to normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Create and train the model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Predicting
y_pred = model_lr.predict(X_test)

# Evaluate performances
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Retrieve the feature names
feature_names = X.columns

# Retrieve the learned coefficients
# Flatten the shape if necessary
coefficients = model_lr.coef_.ravel()

# Create a DataFrame to visualize them
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
})

# Sort by absolute value of the coefficients
coef_df["abs_coef"] = np.abs(coef_df["Coefficient"])
coef_df_sorted = coef_df.sort_values(by="abs_coef", ascending=False)

# Plotting
plt.figure(figsize=(10, 8))
plt.barh(coef_df_sorted["Feature"][:20], coef_df_sorted["Coefficient"][:20])
plt.xlabel("Coefficient")
plt.title("Top 20 features affecting critical temperature")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()

# Show the 10 most influential features
coef_df_sorted.head(10)

# 1. Log transformation of the target
y_log = np.log1p(y)  # log(1 + y), to avoid log(0)

# 2. Train/test split with the transformed target
X_train, X_test, y_train_log, y_test_log = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

# 3. Linear regression on the transformed target
model_log = LinearRegression()
model_log.fit(X_train, y_train_log)

# 4. Prediction and inverse transformation
y_pred_log = model_log.predict(X_test)
y_pred_original = np.expm1(y_pred_log)  # inverse of log1p

# 5. Evaluation in the original scale (Tc)
mse_log = mean_squared_error(y_test, y_pred_original)
r2_log = r2_score(y_test, y_pred_original)

print(f"With log-transformed target:")
print(f"Mean Squared Error: {mse_log:.2f}")
print(f"R² Score: {r2_log:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_original, alpha=0.5, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect prediction line
plt.xlabel("True Tc (K)")
plt.ylabel("Predicted Tc (K)")
plt.title("True vs Predicted Critical Temperature")
plt.grid(True)
plt.tight_layout()
plt.show()
