import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

def load_data(file_path, columns, countries=None, min_history=12, status=None):
    """Load dataset, filter countries, and enforce minimum history."""
    df = pd.read_csv(file_path)
    
    if countries is not None:
        df = df[df['countrycode'].isin(countries)].reset_index(drop=True)
    
    if status is not None:
        status_df = pd.read_csv('all_countries.csv')
        filtered_countries = status_df[status_df['status'].isin(status)]['countrycode'].tolist()
        df = df[df['countrycode'].isin(filtered_countries)].reset_index(drop=True)

    df = df[['countrycode', 'year'] + columns]
    
    # Enforce minimum history per country
    df = enforce_minimum_history(df, min_years=min_history)
    
    return df

def enforce_minimum_history(df, min_years=12):
    """Remove early years for countries without enough history, add years_since_start."""
    df = df.sort_values(['countrycode', 'year'])
    
    def drop_early(g):
        if len(g) < min_years:
            return pd.DataFrame()  # Drop country entirely
        g = g.iloc[min_years:]
        g['years_since_start'] = g['year'] - g['year'].min()
        return g
    
    df_filtered = df.groupby('countrycode', group_keys=False).apply(drop_early).reset_index(drop=True)
    return df_filtered

def create_temporal_features(df, var_name, periods=[1,2,3,5,10]):
    df = df.sort_values(['countrycode', 'year'])
    for period in periods:
        col_name = f'{var_name}_pct_change_{period}y'
        df[col_name] = df.groupby('countrycode')[var_name].pct_change(periods=period, fill_method=None) * 100
    return df

def create_acceleration_features(df, var_name):
    df = df.sort_values(['countrycode', 'year'])
    growth_col = f'{var_name}_pct_change_1y'
    if growth_col in df.columns:
        df[f'{var_name}_acceleration'] = df.groupby('countrycode')[growth_col].diff()
    return df

def create_relative_to_history_features(df, var_name, windows=[3,5,10]):
    df = df.sort_values(['countrycode', 'year'])
    for window in windows:
        rolling_mean = df.groupby('countrycode')[var_name].transform(
            lambda x: x.rolling(window=window, min_periods=max(2, window//2)).mean()
        )
        rolling_std = df.groupby('countrycode')[var_name].transform(
            lambda x: x.rolling(window=window, min_periods=max(2, window//2)).std()
        )
        df[f'{var_name}_zscore_{window}y'] = (df[var_name] - rolling_mean) / rolling_std
    return df

def create_cross_country_features(df, var_name):
    global_mean = df.groupby('year')[var_name].transform('mean')
    df[f'{var_name}_vs_global'] = df[var_name] - global_mean
    df[f'{var_name}_percentile'] = df.groupby('year')[var_name].rank(pct=True) * 100
    return df

def create_lagged_features(df, var_name, lags=[1,2,3]):
    df = df.sort_values(['countrycode', 'year'])
    for lag in lags:
        df[f'{var_name}_lag{lag}'] = df.groupby('countrycode')[var_name].shift(lag)
    return df

def create_all_features(df, base_vars=['gdp','avg_hours','hc'], remove_originals=False, recessions=None, use_recession_file=False):
    df = df.copy().sort_values(['countrycode','year']).reset_index(drop=True)
    print("Creating features...")
    
    for var in base_vars:
        if var not in df.columns:
            print(f"Warning: {var} not found in dataframe")
            continue
        print(f"  Processing {var}...")
        df = create_temporal_features(df, var, periods=[1,2,3,5,10])
        df = create_acceleration_features(df, var)
        df = create_relative_to_history_features(df, var, windows=[3,5,10])
        df = create_cross_country_features(df, var)
        df = create_lagged_features(df, var, lags=[1,2])

    df = add_recession_classification(df, recessions=recessions, use_file=use_recession_file)
    if remove_originals:
        df = df.drop(columns=base_vars)
    print("Feature engineering complete!")
    print(f"Total columns: {len(df.columns)}")
    return df

def add_status_one_hot(df, status_file='all_countries.csv', drop_original=True):
    """
    Merge country status and add one-hot encoded status features.
    """
    status_df = pd.read_csv(status_file)
    status_df = status_df[['countrycode', 'status']]

    # Merge into main dataframe
    df = df.merge(status_df, on='countrycode', how='left')

    # One-hot encode status
    status_dummies = pd.get_dummies(df['status'], prefix='status')

    # Add to dataframe
    df = pd.concat([df, status_dummies], axis=1)

    # Optionally drop original categorical column
    if drop_original:
        df = df.drop(columns=['status'])

    return df


def add_recession_classification(df, recessions=None, use_file=False, future_windows=[1, 2, 3, 5, 10]):
    if use_file and recessions is not None:
        recession_df = pd.read_csv(recessions)
        recession_df = recession_df[['year','country_code']]
        recession_df['recession'] = 1
        df = df.merge(recession_df, left_on=['year','countrycode'], right_on=['year','country_code'], how='left')
        df['recession'] = df['recession'].fillna(0).astype(int)
    else:
        df['recession'] = (df['rgdpe_pct_change_1y'] < 0).astype(int)
    df['recession_last_year'] = df['recession'].shift(1).fillna(0).astype(int)

    df = add_future_recession_window(df, windows=future_windows)

    return df

def add_future_recession_window(df, windows=[1, 2, 3, 5, 10]):
    """
    Adds features like:
    recession_in_3y = 1 if recession occurs in next 3 years, else 0
    """
    df = df.sort_values(['countrycode', 'year']).copy()

    for w in windows:
        col_name = f'recession_in_{w}y'
        df[col_name] = (
            df.groupby('countrycode')['recession']
              .transform(lambda x: x.shift(-1).rolling(w, min_periods=1).max())
              .fillna(0)
              .astype(int)
        )

    return df

def get_feature_columns(df, exclude_cols=['countrycode','year','recession']):
    return [col for col in df.columns if col not in exclude_cols]

def scale_features(df, feature_cols, scaler=None):
    """Scale features using provided scaler or create new one."""
    from sklearn.preprocessing import StandardScaler
    
    if scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    
    return df, scaler

def handle_missing_data(df_features, feature_cols):
    """Handle missing data in engineered features."""
    # Forward fill lagged features only
    lag_cols = [col for col in df_features.columns if '_lag' in col]
    df_features[lag_cols] = df_features.groupby('countrycode')[lag_cols].fillna(method='ffill')

    # Fill rolling statistics with 0 or median
    rolling_cols = [col for col in df_features.columns if '_zscore_' in col or '_vs_global' in col]
    df_features[rolling_cols] = df_features[rolling_cols].fillna(0)

    # Only drop rows where critical features are missing
    critical_cols = [col for col in feature_cols if '_pct_change_1y' in col]
    df_features = df_features.dropna(subset=critical_cols)

    # Drop any remaining NaNs
    df_features = df_features.dropna().reset_index(drop=True)
    return df_features

if __name__ == "__main__":
    use_all_countries = True
    if use_all_countries:
        countries_to_use = None  # Use all countries in the dataset
    else:
        countries_to_use = ["USA","DNK","NLD","GBR","JPN","CAN","AUS","EGY","BRA","CHN"]

    columns_to_use = ['rgdpe', 'emp', 'ck', 'csh_i', 'rtfpna', 'csh_c']

    use_recession_file = True  # Whether to use external recession data for extra test set
    recession_file = 'recessions.csv'  # Path to recession data file if used
    
    # Load data with minimum history enforced
    df = load_data('cleaned_V11.csv', countries=countries_to_use, columns=columns_to_use, min_history=12, status=['Developed', 'Developing', 'Least Developed'])
    
    # Create features
    df_features = create_all_features(df, base_vars=columns_to_use, remove_originals=False)
    
    # Get feature columns
    feature_cols = get_feature_columns(df_features, exclude_cols=['countrycode', 'year', 'years_since_start', 'recession_next_year', 'recession', 'recession_last_year', 'recession_in_1y', 'recession_in_2y', 'recession_in_3y', 'recession_in_5y', 'recession_in_10y'])

    df_features = add_status_one_hot(df_features, status_file='all_countries.csv')
    
    # Handle missing data
    df_features = handle_missing_data(df_features, feature_cols)
    
    # Scale features and save scaler
    df_features, fitted_scaler = scale_features(df_features, feature_cols)

    # Save the scaler for later use
    joblib.dump(fitted_scaler, 'feature_scaler.pkl')
    print("Scaler saved to 'feature_scaler.pkl'")
    
    # # Also save feature column names for reference
    # joblib.dump(feature_cols, 'feature_columns.pkl')
    # print("Feature column names saved to 'feature_columns.pkl'")

    # Drop any remaining NaNs
    # df_features = df_features.dropna().reset_index(drop=True)
    
    # Save main dataset
    df_features.to_csv('engineered_features_all_countries.csv', index=False)
    print(f"Main dataset saved with {len(df_features)} rows")
    
    # If using recession file, create separate recession dataset with same scaling
    if use_recession_file:
        print("\nProcessing recession data separately...")
        recession_df = pd.read_csv(recession_file)

        # Load data for recession countries only
        df_recession = load_data('cleaned_V11.csv', 
                                countries=recession_df['country_code'].unique().tolist(), 
                                columns=columns_to_use, 
                                min_history=12)
        
        # Create features
        df_recession_features = create_all_features(df_recession, 
                                                    base_vars=columns_to_use, 
                                                    remove_originals=False, 
                                                    recessions=recession_file, 
                                                    use_recession_file=True)
        
        # Apply the SAME scaler that was fitted on the main data
        df_recession_features, _ = scale_features(df_recession_features, feature_cols, scaler=fitted_scaler)
        
        # Handle missing data
        df_recession_features = handle_missing_data(df_recession_features, feature_cols)
        
        # Save recession dataset
        df_recession_features.to_csv('test_set.csv', index=False)
        print(f"Recession dataset saved with {len(df_recession_features)} rows")
        print(f"Recession years: {df_recession_features['recession'].sum()}")