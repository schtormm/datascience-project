import pandas as pd

def load_data(file_path, countries, columns):
    """Load dataset and filter for specified countries."""
    df = pd.read_csv(file_path)
    df_filtered = df[df['countrycode'].isin(countries)].reset_index(drop=True)
    df_filtered = df_filtered[['countrycode', 'year'] + columns]
    return df_filtered

def create_pct_change_features(df, columns, years=[1, 2, 3, 5, 10]):
    """Create percentage change features for specified columns over given years."""
    for col in columns:
        for year in years:
            df[f'{col}_pct_change_{year}yr'] = df[col].pct_change(periods=year) * 100

    # Drop rows with NaN values resulting from percentage change calculations
    df = df.dropna().reset_index(drop=True)
    return df

def normalize_values(df, columns):
    """Normalize specified columns using min-max scaling."""
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

def standardize_values(df, columns):
    """Standardize specified columns to have mean 0 and standard deviation 1."""
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[col] = (df[col] - mean_val) / std_val
    return df

def save_data(df, file_path):
    """Save the processed DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    countries_to_use = ["USA", "DNK", "NLD"]
    columns_to_use = ['rgdpe', 'emp', 'hc']
    columns_to_transform = ['rgdpe', 'emp', 'hc']
    
    # Load the dataset
    data = load_data('cleaned_V11.csv', countries_to_use, columns_to_use)

    # Create percentage change features
    data_with_pct_changes = create_pct_change_features(data, columns_to_transform)

    save_data(data_with_pct_changes, 'data_with_pct_changes.csv')