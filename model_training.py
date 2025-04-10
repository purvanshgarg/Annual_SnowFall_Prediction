import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)
from scipy import stats

def train_sarimax_model(X_train, y_train):
    """
    Train a SARIMAX model with exogenous variables.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
    
    Returns:
        SARIMAX model results
    """
    # Prepare numeric features and clean target variable
    X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
    X_train_numeric_array = np.asarray(X_train_numeric)
    
    # Clean target variable
    y_train_cleaned = _clean_target_variable(y_train)
    
    # Ensure feature and target lengths match
    X_train_numeric_array = X_train_numeric_array[-len(y_train_cleaned):]
    
    # Define and fit SARIMAX model
    model = SARIMAX(
        y_train_cleaned,
        exog=X_train_numeric_array,
        order=(2, 0, 2),
        seasonal_order=(2, 0, 2, 3),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    return model.fit(disp=False)

def _clean_target_variable(y_train, threshold=1e-8):
    """
    Clean target variable by replacing near-zero values.
    
    Args:
        y_train (pd.Series): Original target series
        threshold (float): Threshold for near-zero values
    
    Returns:
        pd.Series: Cleaned target series
    """
    non_zero_mean = y_train[abs(y_train) > threshold].mean()
    y_train_cleaned = y_train.copy()
    y_train_cleaned.loc[abs(y_train_cleaned) <= threshold] = non_zero_mean
    
    return y_train_cleaned

def evaluate_forecast(y_true, y_pred):
    """
    Comprehensive forecast performance evaluation.
    
    Args:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values
    
    Returns:
        dict: Performance metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'Mean Squared Error (MSE)': mean_squared_error(y_true, y_pred),
        'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y_true, y_pred)),
        'Mean Absolute Error (MAE)': mean_absolute_error(y_true, y_pred),
        'R-squared (R²)': r2_score(y_true, y_pred)
    }
    
    try:
        metrics['Mean Absolute Percentage Error (MAPE)'] = _custom_mape(y_true, y_pred)
    except Exception:
        metrics['Mean Absolute Percentage Error (MAPE)'] = np.nan
    
    # Residual analysis
    residuals = y_true - y_pred
    metrics['Mean of Residuals'] = np.mean(residuals)
    metrics['Standard Deviation of Residuals'] = np.std(residuals)
    
    try:
        metrics['Skewness of Residuals'] = stats.skew(residuals)
        metrics['Kurtosis of Residuals'] = stats.kurtosis(residuals)
    except Exception:
        metrics['Skewness of Residuals'] = np.nan
        metrics['Kurtosis of Residuals'] = np.nan
    
    return metrics

def _custom_mape(y_true, y_pred):
    """
    Custom Mean Absolute Percentage Error calculation.
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
    
    Returns:
        float: Mean Absolute Percentage Error
    """
    mask = y_true != 0
    percentage_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
    return np.mean(percentage_errors)
