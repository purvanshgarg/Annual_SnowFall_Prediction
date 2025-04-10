import os
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_data
from model_preparation import prepare_seasonal_data
from model_training import train_sarimax_model, evaluate_forecast
from exploratory_data_analysis import perform_eda

def main():
    """
    Main execution script for the meteorological forecasting pipeline.
    
    This script integrates:
    1. Data loading and preprocessing
    2. Exploratory data analysis (EDA)
    3. Seasonal data preparation
    4. SARIMAX model training
    5. Forecast generation and evaluation
    """
    # Create output directory for plots
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("Snowfall Prediction using ERA5 data ")
    print("="*80)
    
    # Step 1: Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    df = load_and_preprocess_data("merged_raw_all.csv") #may need to change according to file location
    print(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Step 2: Perform exploratory data analysis
    print("\n[2/5] Performing exploratory data analysis...")
    eda_results = perform_eda(df, target_column='msr', save_path=output_dir)
    
    # Print some key EDA insights
    print(f"Basic statistics for target variable:")
    for stat, value in eda_results['basic_stats'].items():
        print(f"  - {stat}: {value:.6f}")
    
    print("\nStationarity check:")
    if eda_results['stationarity']['is_stationary']:
        print("  - Time series is stationary (p-value < 0.05)")
    else:
        print("  - Time series is not stationary (p-value >= 0.05)")
        if 'diff1_is_stationary' in eda_results['stationarity'] and eda_results['stationarity']['diff1_is_stationary']:
            print("  - First difference is stationary")
    
    print("\nAnomaly detection:")
    if 'anomaly_count' in eda_results['decomposition']:
        print(f"  - {eda_results['decomposition']['anomaly_count']} anomalies detected ({eda_results['decomposition']['anomaly_percentage']:.2f}%)")
    
    print("\nTop correlations with target variable:")
    for feature, corr in eda_results['top_correlations'].items():
        print(f"  - {feature}: {corr:.4f}")
    
    # Step 3: Prepare seasonal data
    print("\n[3/5] Preparing seasonal data for modeling...")
    X_train_scaled, X_test_scaled, y_train, y_test = prepare_seasonal_data(df)
    print(f"Training data: {X_train_scaled.shape[0]} samples")
    print(f"Testing data: {X_test_scaled.shape[0]} samples")
    
    # Step 4: Train SARIMAX model
    print("\n[4/5] Training SARIMAX model...")
    results = train_sarimax_model(X_train_scaled, y_train)
    print("Model training complete")
    print("\nModel summary:")
    print(results.summary().tables[0].as_text())
    print("\nAIC:", results.aic)
    print("BIC:", results.bic)
    
    # Step 5: Generate forecast and evaluate
    print("\n[5/5] Generating and evaluating forecast...")
    forecast_start = pd.Timestamp('2023-10-01')
    forecast_end = pd.Timestamp('2024-02-29')
    forecast_index = pd.date_range(start=forecast_start, end=forecast_end, freq='6h')
    
    forecast = results.get_forecast(steps=len(forecast_index), exog=X_test_scaled)
    conf_int_values = forecast.conf_int(alpha=0.05)[:len(forecast_index)]
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # First subplot: Full time series
    plt.subplot(2, 1, 1)
    plt.plot(y_train.index, y_train, label='Training Data', color='blue', alpha=0.7)
    plt.plot(y_test.index, y_test, label='Actual Test Data', color='green', alpha=0.7)
    plt.plot(forecast_index, forecast.predicted_mean, label='Forecast', color='red', linestyle='--')
    plt.title("Time Series Forecast with SARIMAX Model")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Second subplot: Forecast with confidence interval
    plt.subplot(2, 1, 2)
    plt.plot(forecast_index, forecast.predicted_mean, label='Forecast Mean', color='red', linestyle='--')
    plt.fill_between(forecast_index, 
                     conf_int_values['lower msr'], 
                     conf_int_values['upper msr'], 
                     color='black', alpha=0.3, 
                     label='95% Confidence Interval')
    plt.plot(y_test.index, y_test, label='Actual Values', color='green', alpha=0.7)
    plt.title("Forecast with Confidence Interval vs Actual Values")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save forecast plot
    plt.savefig(f"{output_dir}/forecast_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Evaluate forecast
    metrics = evaluate_forecast(y_test, forecast.predicted_mean)
    print("\nForecast Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    print("\nAnalysis complete. Results and plots saved to the 'output' directory.")

if __name__ == "__main__":
    main()
