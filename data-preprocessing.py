import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path, columns, countries=None, min_history=12):
    """Load dataset, filter countries, and enforce minimum history."""
    df = pd.read_csv(file_path)
    
    if countries is not None:
        df = df[df['countrycode'].isin(countries)].reset_index(drop=True)
    
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
        df[col_name] = df.groupby('countrycode')[var_name].pct_change(periods=period) * 100
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

def add_recession_classification(df, recessions=None, use_file=False):
    if use_file and recessions is not None:
        recession_df = pd.read_csv(recessions)
        recession_df = recession_df[['year','country_code']]
        recession_df['recession'] = 1
        df = df.merge(recession_df, left_on=['year','countrycode'], right_on=['year','country_code'], how='left')
        df['recession'] = df['recession'].fillna(0).astype(int)
        df = df.drop(columns=['country_code'])
    else:
        df['recession'] = (df['rgdpe_pct_change_1y'] < 0).astype(int)
    df['recession_last_year'] = df['recession'].shift(1).fillna(0).astype(int)
    df['recession_next_year'] = df['recession'].shift(-1).fillna(0).astype(int)
    return df

def get_feature_columns(df, exclude_cols=['countrycode','year','recession']):
    return [col for col in df.columns if col not in exclude_cols]

def scale_features(df, feature_cols):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df


if __name__ == "__main__":
    countries_to_use = ["USA","DNK","NLD","GBR","JPN","CAN","AUS","EGY","BRA","CHN"]
    columns_to_use = ['rgdpe']
    use_recession_file = False  # Set to True if using external recession data
    recession_file = 'recessions.csv'  # Path to recession data file if used
    
    # Load data with minimum history enforced
    df = load_data('cleaned_V11.csv', columns=columns_to_use, min_history=12)
    
    # Create features
    df_features = create_all_features(df, base_vars=columns_to_use, remove_originals=True, recessions=recession_file, use_recession_file=use_recession_file)
    
    # Get feature columns
    feature_cols = get_feature_columns(df_features, exclude_cols=['countrycode', 'year', 'years_since_start', 'recession_next_year', 'recession', 'recession_last_year'])
    
    # Scale features
    df_features = scale_features(df_features, feature_cols)

    # Drop any remaining NaNs
    df_features = df_features.dropna().reset_index(drop=True)
    
    # Save for inspection
    df_features.to_csv('engineered_features_all_countries.csv', index=False)
