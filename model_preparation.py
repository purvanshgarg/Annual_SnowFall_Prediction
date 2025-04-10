import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_seasonal_data(df, target_column='msr', train_years=range(2020, 2023), test_years=[2023]):
    """
    Prepare seasonal data for model training and testing.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target variable column
        train_years (list): Years to use for training
        test_years (list): Years to use for testing
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test)
    """
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Extract seasonal data
    X_train, y_train = _extract_seasonal_data(X, y, train_years)
    X_test, y_test = _extract_seasonal_data(X, y, test_years)

    # Identify numeric columns for scaling
    numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])

    # Optional: Remove 'valid_time' column from test data
    if 'valid_time' in X_test_scaled.columns:
        X_test_scaled.drop('valid_time', axis=1, inplace=True)

    return X_train_scaled, X_test_scaled, y_train, y_test

def _extract_seasonal_data(X, y, years):
    """
    Extract seasonal data for specified years with 6-hour frequency.
    
    Args:
        X (pd.DataFrame): Feature data
        y (pd.Series): Target data
        years (list): Years to extract
    
    Returns:
        tuple: (X_seasonal, y_seasonal)
    """
    X_seasonal_list = []
    y_seasonal_list = []
    
    for year in years:
        # Define start and end for the season
        start_date = pd.Timestamp(f'{year}-10-01')
        try:
            end_date = pd.Timestamp(f'{year+1}-02-29')
        except ValueError:
            end_date = pd.Timestamp(f'{year+1}-02-28')
        
        # Create full 6-hour frequency index
        full_index = pd.date_range(start=start_date, end=end_date, freq='6H')
        
        # Reindex X and y
        X_period = X.loc[start_date:end_date].reindex(full_index)
        y_period = y.loc[start_date:end_date].reindex(full_index)
        
        # Forward fill missing values
        X_period = X_period.fillna(method='ffill')
        y_period = y_period.fillna(method='ffill')
        
        X_seasonal_list.append(X_period)
        y_seasonal_list.append(y_period)
    
    # Concatenate seasonal data
    X_seasonal = pd.concat(X_seasonal_list)
    y_seasonal = pd.concat(y_seasonal_list)
    
    # Set frequency
    X_seasonal.index.freq = pd.infer_freq(X_seasonal.index)
    y_seasonal.index.freq = pd.infer_freq(y_seasonal.index)
    
    return X_seasonal, y_seasonal
