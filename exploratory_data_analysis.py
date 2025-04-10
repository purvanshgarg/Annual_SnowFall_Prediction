import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def perform_eda(df, target_column='msr', save_path=None):
    """
    Perform comprehensive exploratory data analysis on meteorological data.
    
    Args:
        df (pd.DataFrame): The input dataframe with timestamp index
        target_column (str): The name of the target variable column (default: 'msr')
        save_path (str): Directory to save plots (default: None, plots are displayed)
    
    Returns:
        dict: Dictionary containing EDA results and statistics
    """
    results = {}
    
    # Check if dataframe has datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        if 'valid_time' in df.columns:
            df['valid_time'] = pd.to_datetime(df['valid_time'])
            df.set_index('valid_time', inplace=True)
        else:
            raise ValueError("DataFrame must have datetime index or 'valid_time' column")
    
    # Extract temporal features
    df_features = _extract_temporal_features(df)
    results['temporal_features'] = df_features.columns.tolist()
    
    # Calculate wind features if components exist
    if all(col in df.columns for col in ['u100', 'v100', 'u10n', 'v10n']):
        df_wind = _calculate_wind_features(df)
        results['wind_features'] = df_wind.columns.tolist()
    
    # Basic statistics
    results['basic_stats'] = df[target_column].describe().to_dict()
    results['missing_values'] = df[target_column].isna().sum()
    
    # Distribution of target variable
    _plot_distribution(df, target_column, save_path)
    
    # Time series visualization
    _plot_time_series(df, target_column, save_path)
    
    # Seasonal patterns
    years = sorted(df.index.year.unique())
    _plot_seasonal_data(df, target_column, years, save_path)
    
    # Correlation analysis
    correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
    top_correlations = _get_top_correlations(correlation_matrix, target_column)
    results['top_correlations'] = top_correlations
    _plot_correlation_heatmap(correlation_matrix, save_path)
    
    # Decomposition and anomaly detection
    decomposition_results = _perform_time_series_decomposition(df, target_column, save_path)
    results['decomposition'] = decomposition_results
    
    # Stationarity check
    stationarity_results = _check_stationarity(df[target_column], save_path)
    results['stationarity'] = stationarity_results
    
    # ACF and PACF analysis for ARIMA order selection
    _plot_acf_pacf(df[target_column], save_path)
    
    return results


def _extract_temporal_features(df):
    """Extract temporal features from datetime index."""
    df = df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week
    df['season'] = df['month'].apply(_get_season)
    
    return df


def _get_season(month):
    """Map month to meteorological season."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


def _calculate_wind_features(df):
    """Calculate wind speed and direction from u and v components."""
    df = df.copy()
    
    # Calculate wind speed
    df['wind_speed_100m'] = np.sqrt(df['u100']**2 + df['v100']**2)
    df['wind_speed_10m'] = np.sqrt(df['u10n']**2 + df['v10n']**2)
    
    # Calculate wind direction in degrees
    df['wind_dir_100m'] = np.degrees(np.arctan2(df['v100'], df['u100'])) % 360
    df['wind_dir_10m'] = np.degrees(np.arctan2(df['v10n'], df['u10n'])) % 360
    
    return df


def _plot_distribution(df, target_column, save_path=None):
    """Plot distribution of target variable."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_column].dropna(), bins=50, kde=True, color='#3498db')
    plt.title(f'Distribution of {target_column.capitalize()}', fontsize=14)
    plt.xlabel(f'{target_column.capitalize()} Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(f"{save_path}/distribution_plot.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
    # Additional stats plot
    plt.figure(figsize=(10, 6))
    df[target_column].dropna().plot(kind='box', vert=False, color='#3498db')
    plt.title(f'Boxplot of {target_column.capitalize()}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(f"{save_path}/boxplot.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def _plot_time_series(df, target_column, save_path=None):
    """Plot time series of target variable."""
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df[target_column], color='#3498db', linewidth=1)
    plt.title(f'Time Series of {target_column.capitalize()}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'{target_column.capitalize()} Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add monthly averages
    monthly_avg = df[target_column].resample('M').mean()
    plt.plot(monthly_avg.index, monthly_avg.values, color='#e74c3c', 
             linewidth=2, linestyle='--', label='Monthly Average')
    plt.legend()
    
    if save_path:
        plt.savefig(f"{save_path}/time_series_plot.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def _plot_seasonal_data(df, target_column, years, save_path=None):
    """Plot seasonal data for specified years from Oct to Feb."""
    plt.figure(figsize=(14, 7))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    for i, year in enumerate(years):
        # Define winter season (Oct to Feb of next year)
        start_date = pd.Timestamp(f'{year}-10-01')
        try:
            end_date = pd.Timestamp(f'{year+1}-02-29')  # Check for leap year
        except ValueError:
            end_date = pd.Timestamp(f'{year+1}-02-28')
            
        # Filter data for current season
        season_data = df.loc[start_date:end_date, target_column]
        
        if not season_data.empty:
            color_idx = i % len(colors)
            plt.plot(season_data.index, season_data, color=colors[color_idx],
                     linewidth=1.5, label=f'Winter {year}-{year+1}')
    
    plt.title('Seasonal Comparison (Oct - Feb)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'{target_column.capitalize()} Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(f"{save_path}/seasonal_comparison.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
    # Monthly average by year
    plt.figure(figsize=(14, 7))
    monthly_avg = df.groupby([df.index.year, df.index.month])[target_column].mean().unstack()
    monthly_avg.plot(kind='line', marker='o')
    plt.title('Monthly Average by Year', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel(f'Average {target_column.capitalize()}', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(f"{save_path}/monthly_avg_by_year.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def _get_top_correlations(correlation_matrix, target_column, n=10):
    """Get top correlations with target variable."""
    correlations = correlation_matrix[target_column].sort_values(ascending=False)
    # Remove self-correlation
    correlations = correlations[correlations.index != target_column]
    return correlations.head(n).to_dict()


def _plot_correlation_heatmap(correlation_matrix, save_path=None):
    """Plot correlation heatmap."""
    plt.figure(figsize=(12, 10))
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=False, mask=mask, cmap='coolwarm',
                vmin=-1, vmax=1, center=0, linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=14)
    
    if save_path:
        plt.savefig(f"{save_path}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def _perform_time_series_decomposition(df, target_column, save_path=None):
    """Perform STL decomposition and anomaly detection."""
    results = {}
    
    # Ensure the series has a regular frequency
    series = df[target_column].copy()
    
    # Handle missing values for decomposition
    series = series.dropna()
    
    try:
        # Determine period based on data frequency
        if isinstance(df.index.freq, pd.tseries.offsets.BaseOffset):
            freq_str = df.index.freq.name
            if freq_str in ['H', '6H']:
                # For hourly data (24*7 = 168 hours per week, or 28 observations per week for 6H data)
                period = 28 if freq_str == '6H' else 168
            elif freq_str == 'D':
                # For daily data (365.25/12 ≈ 30 days per month)
                period = 30
            else:
                # Default to 4 observations per cycle
                period = 4
        else:
            # If frequency is not set, infer from average difference between timestamps
            avg_hours = (df.index[1:] - df.index[:-1]).mean().total_seconds() / 3600
            if 5 <= avg_hours <= 7:  # Around 6 hours
                period = 28  # 4 observations per day * 7 days
            elif 23 <= avg_hours <= 25:  # Around 24 hours
                period = 30  # 30 days per month
            else:
                period = 4  # Default
        
        # Perform STL decomposition
        stl = STL(series, period=period)
        result = stl.fit()
        
        # Extract components
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid
        
        # Store results
        results['trend_mean'] = trend.mean()
        results['seasonal_mean'] = seasonal.mean()
        results['residual_mean'] = residual.mean()
        results['trend_std'] = trend.std()
        results['seasonal_std'] = seasonal.std()
        results['residual_std'] = residual.std()
        
        # Plot decomposition
        plt.figure(figsize=(14, 10))
        result.plot()
        plt.suptitle('STL Decomposition', fontsize=14)
        
        if save_path:
            plt.savefig(f"{save_path}/stl_decomposition.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
        # Anomaly detection using IQR on residuals
        Q1, Q3 = np.percentile(residual.dropna(), [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Detect anomalies
        anomalies = residual[(residual < lower_bound) | (residual > upper_bound)]
        
        # Store anomaly results
        results['anomaly_count'] = len(anomalies)
        results['anomaly_percentage'] = 100 * len(anomalies) / len(residual)
        
        # Plot anomalies
        plt.figure(figsize=(14, 7))
        plt.plot(residual, color='#3498db', linestyle='-', linewidth=1, label='Residuals')
        plt.scatter(anomalies.index, anomalies, color='#e74c3c', label='Anomalies')
        plt.axhline(y=upper_bound, color='#f39c12', linestyle='--', label=f'Upper bound: {upper_bound:.2f}')
        plt.axhline(y=lower_bound, color='#f39c12', linestyle='--', label=f'Lower bound: {lower_bound:.2f}')
        plt.title('Anomaly Detection using IQR', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Residual Value', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(f"{save_path}/anomaly_detection.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    except Exception as e:
        results['error'] = str(e)
        print(f"Error in time series decomposition: {e}")
    
    return results


def _check_stationarity(series, save_path=None):
    """Check stationarity using Augmented Dickey-Fuller test."""
    results = {}
    
    series = series.dropna()
    
    # Perform ADF test
    adf_result = adfuller(series)
    
    # Store results
    results['adf_statistic'] = adf_result[0]
    results['p_value'] = adf_result[1]
    results['critical_values'] = adf_result[4]
    results['is_stationary'] = adf_result[1] < 0.05
    
    # If not stationary, check first difference
    if not results['is_stationary']:
        series_diff = series.diff().dropna()
        adf_result_diff = adfuller(series_diff)
        
        results['diff1_adf_statistic'] = adf_result_diff[0]
        results['diff1_p_value'] = adf_result_diff[1]
        results['diff1_is_stationary'] = adf_result_diff[1] < 0.05
        
        # Plot original vs differenced series
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Original series
        ax1.plot(series, color='#3498db')
        ax1.set_title('Original Series', fontsize=14)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Differenced series
        ax2.plot(series_diff, color='#2ecc71')
        ax2.set_title('First Differenced Series', fontsize=14)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Value', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/stationarity_check.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    return results


def _plot_acf_pacf(series, save_path=None, lags=40):
    """Plot ACF and PACF for ARIMA order selection."""
    series = series.dropna()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ACF plot
    plot_acf(series, lags=lags, ax=ax1)
    ax1.set_title('Autocorrelation Function (ACF)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # PACF plot
    plot_pacf(series, lags=lags, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function (PACF)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/acf_pacf_plots.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
    # If not stationary, also plot ACF/PACF of differenced series
    adf_result = adfuller(series)
    if adf_result[1] >= 0.05:
        series_diff = series.diff().dropna()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # ACF plot of differenced series
        plot_acf(series_diff, lags=lags, ax=ax1)
        ax1.set_title('ACF of Differenced Series', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # PACF plot of differenced series
        plot_pacf(series_diff, lags=lags, ax=ax2)
        ax2.set_title('PACF of Differenced Series', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/acf_pacf_diff_plots.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
