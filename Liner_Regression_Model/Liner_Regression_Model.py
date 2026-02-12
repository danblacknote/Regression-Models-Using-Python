"""
Maternal Hemoglobin at Delivery - Linear Regression Analysis
=============================================================
This script performs linear regression analysis on maternal hemoglobin levels
at delivery, predicting Z-Scores from raw hemoglobin values.

Author: [Your Name]
Date: [Current Date]
"""

# ============================================================================
# 1. LIBRARY IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import scipy.stats as stats
import pylab
import math

# ============================================================================
# 2. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================
# Load the maternal hemoglobin dataset from Excel file
df = pd.read_excel('E:/All Dan Document/Stastics and percentage calculation APHI/Final/Final Data Analysis with Diffrent Parameter/Maternal Hemoglobin at Delivery.xlsx')

# Display first 5 rows to verify data loading
print("=== FIRST 5 ROWS OF DATASET ===")
print(df.head())
print("\n")

# Display statistical summary of numerical columns
print("=== DESCRIPTIVE STATISTICS ===")
print(df.describe())
print("\n")

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
# Create a pairplot to visualize relationships between all variables
# Alpha=0.5 adds transparency to handle overlapping points
sns.pairplot(df, kind='scatter', plot_kws={'alpha': 0.5})
plt.suptitle('Pairplot of Maternal Hemoglobin Variables', y=1.02)
plt.show()

# ============================================================================
# 4. DATA PREPARATION FOR MODELING
# ============================================================================
# Define features (X) and target variable (y)
# NOTE: Predicting Z-Score from the result creates a deterministic relationship
# This is for demonstration purposes only - not a valid prediction task
X = df[['Result']]      # Feature: Raw hemoglobin value
y = df['Z-Score']       # Target: Standardized Z-Score

# Split data into training (70%) and testing (30%) sets
# random_state=0 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=0
)

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
# Initialize and train Linear Regression model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Display model parameters
print("=== MODEL PARAMETERS ===")
print(f"Coefficient: {lm.coef_[0]:.4f}")
print(f"Intercept: {lm.intercept_:.4f}")
print("\n")

# ============================================================================
# 6. MODEL PREDICTION
# ============================================================================
# Generate predictions on the test set
y_predict = lm.predict(X_test)
print("=== SAMPLE PREDICTIONS (First 5) ===")
print(f"Predicted Z-Scores: {y_predict[:5]}")
print(f"Actual Z-Scores: {y_test[:5].values}")
print("\n")

# ============================================================================
# 7. MODEL EVALUATION - VISUALIZATION
# ============================================================================
"""
Predicted vs Actual Values Plot
-------------------------------
This scatter plot compares predicted Z-Scores against actual Z-Scores.
In a valid model, points should cluster tightly around the diagonal line.
Note: Due to the deterministic relationship between Result and Z-Score,
this plot shows an artificially perfect linear relationship.
"""
sns.scatterplot(x=y_predict, y=y_test)
plt.xlabel('Predicted Z-Score')
plt.ylabel('Actual Z-Score')
plt.title('Predicted vs Actual Z-Score Values')
plt.axline([0, 0], [1, 1], color='red', linestyle='--', label='Perfect Prediction')
plt.legend()
plt.show()

# ============================================================================
# 8. MODEL EVALUATION - METRICS
# ============================================================================
# Calculate R-squared (coefficient of determination)
# Range: 0-1, higher values indicate better fit
r_score = r2_score(y_test, y_predict)  # Note: Order is (y_true, y_pred)
print("=== MODEL PERFORMANCE METRICS ===")
print(f"R² Score: {r_score:.4f}")

# Calculate error metrics
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = math.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print("\n")

# ============================================================================
# 9. RESIDUAL ANALYSIS
# ============================================================================
"""
Residuals = Actual - Predicted values
Residual analysis helps validate linear regression assumptions:
1. Normality of residuals
2. Homoscedasticity (constant variance)
3. Independence
"""
residuals = y_test - y_predict
print("=== RESIDUAL ANALYSIS ===")
print(f"Residual Summary:")
print(f"  Mean: {residuals.mean():.4f}")
print(f"  Std: {residuals.std():.4f}")
print(f"  Min: {residuals.min():.4f}")
print(f"  Max: {residuals.max():.4f}")

# 9.1 Residual Distribution Plot
# Checks if residuals are normally distributed (bell-shaped curve)
sns.displot(residuals, bins=20, kde=True)
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()

# 9.2 Q-Q Plot (Quantile-Quantile Plot)
# Tests normality assumption - points should follow the diagonal line
stats.probplot(residuals, dist="norm", plot=pylab)
plt.title('Q-Q Plot of Residuals')
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# 10. OUTLIER DETECTION
# ============================================================================
def detect_outliers_zscore(dataframe, column, threshold=3):
    """
    Detect outliers in a column using the Z-score method.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        The dataframe containing the data
    column: str
        Name of the column to check for outliers
    threshold: float
        Z-score threshold (default: 3)
    
    Returns:
    --------
    list: Values identified as outliers
    """
    outliers = []
    mean = np.mean(dataframe[column])
    std = np.std(dataframe[column])
    
    for value in dataframe[column]:
        z_score = (value - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(value)
    
    return outliers

# Detect outliers in the Result column
outliers = detect_outliers_zscore(df, 'Result', threshold=3)
print("\n=== OUTLIER DETECTION ===")
print(f"Number of outliers detected: {len(outliers)}")
print(f"Outlier values: {outliers}")

# ============================================================================
# 11. OUTLIER VISUALIZATION
# ============================================================================
# Box plot provides visual identification of outliers
# Circles beyond whiskers indicate potential outliers
plt.figure(figsize=(8, 6))
plt.boxplot(df['Result'])
plt.title('Box Plot of Maternal Hemoglobin Results')
plt.ylabel('Hemoglobin Level')
plt.grid(True, alpha=0.3)
plt.show()

print("\n=== ANALYSIS COMPLETE ===")
