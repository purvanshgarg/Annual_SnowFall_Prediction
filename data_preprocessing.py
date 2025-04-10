import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the meteorological dataset.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Column renaming mapping
    column_mapping = {
        'fg10': '10m_wind_gust',
        'd2m': '2m_dewpoint_temperature',
        't2m': '2m_temperature',
        'anor': 'angle_of_sub_gridscale_orography',
        'mser': 'mean_snow_evaporation_rate',
        'avg_tsrwe': 'mean_snowfall_rate',
        'cbh': 'cloud_base_height',
        'mcc': 'medium_cloud_cover',
        'ptype': 'precipitation_type',
        'sp': 'surface_pressure',
        'tcsw': 'total_column_snow_water',
        'tcslw': 'total_column_supercooled_liquid_water',
        'deg0l': 'zero_degree_level',
        # ... other mappings remain the same
    }

    # Load data
    df = pd.read_csv(file_path)
    df = df.drop(columns=['latitude', 'longitude', 'number'])
    df.rename(columns=column_mapping, inplace=True)

    # Convert valid_time to datetime
    df['valid_time'] = pd.to_datetime(df['valid_time'], format='%Y-%m-%d %H:%M:%S')

    # Add 6-hourly period identification
    df['period'] = df['valid_time'].apply(_assign_6hourly_period)

    # Create composite dataset
    df_composite = df.groupby(['period']).apply(_create_composite).reset_index(drop=True)

    # Reverse column mapping for restoration
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    df_composite.rename(columns=reverse_mapping, inplace=True)
    df_composite.rename(columns={'avg_tsrwe': 'msr'}, inplace=True)

    # Set index and sort
    df_composite['period'] = pd.to_datetime(df_composite['period'])
    df_composite.set_index('period', inplace=True)
    df_composite.sort_index(inplace=True)

    return df_composite

def _assign_6hourly_period(timestamp):
    """Assign 6-hourly period to a timestamp."""
    hour = timestamp.hour
    if 0 <= hour < 6:
        return timestamp.replace(hour=0, minute=0, second=0)
    elif 6 <= hour < 12:
        return timestamp.replace(hour=6, minute=0, second=0)
    elif 12 <= hour < 18:
        return timestamp.replace(hour=12, minute=0, second=0)
    else:
        return timestamp.replace(hour=18, minute=0, second=0)

def _create_composite(group):
    """
    Create a composite row for each unique period with proper scientific aggregations.
    
    Args:
        group (pd.DataFrame): Grouped dataframe
    
    Returns:
        pd.Series: Composite row
    """
    # Define aggregation methods
    mean_cols = [
        'wind_speed_10m', 'wind_speed_100m', 'wind_dir_100m', 'wind_dir_10m',
        '2m_dewpoint_temperature', '2m_temperature', 'mean_sea_level_pressure',
        'surface_pressure', 'boundary_layer_height', 'convective_inhibition',
        'friction_velocity', 'geopotential', 
        'total_column_supercooled_liquid_water', 'zero_degree_level',
        'cloud_base_height'
    ]
    
    sum_cols = [
        'mean_snow_evaporation_rate', 'mean_snowfall_rate',
        'convective_snowfall_rate_water_equivalent',
        'large_scale_snowfall_rate_water_equivalent',
        'total_column_snow_water', 'convective_snowfall',
        'large_scale_snowfall', 'boundary_layer_dissipation'
    ]
    
    max_cols = [
        '10m_wind_gust',
        'high_cloud_cover', 'low_cloud_cover', 'medium_cloud_cover'
    ]
    
    # Initialize composite row
    composite_row = group.iloc[0].copy()
    
    # Apply aggregations
    for col in mean_cols:
        if col in group.columns:
            composite_row[col] = group[col].mean()
    
    for col in sum_cols:
        if col in group.columns:
            composite_row[col] = group[col].sum()
    
    for col in max_cols:
        if col in group.columns:
            composite_row[col] = group[col].max()
    
    # Handle categorical columns
    if 'precipitation_type' in group.columns:
        composite_row['precipitation_type'] = (
            group['precipitation_type'].mode()[0] 
            if not group['precipitation_type'].mode().empty 
            else np.nan
        )
    
    # Set the composite time
    composite_row['valid_time'] = group['period'].iloc[0]
    
    return composite_row
