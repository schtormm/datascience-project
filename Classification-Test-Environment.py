import pandas as pd
from sklearn.model_selection import train_test_split

def import_data(file_path, country=None):
    """Import data from a CSV file."""
    df = pd.read_csv(file_path)
    if country is None:
        return df
    else:
        return df[df['countrycode'] == country]

def split_features_labels(df, feature_cols, label_col):
    """Split the DataFrame into features and labels."""
    X = df[feature_cols]
    y = df[label_col]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_model(X_train, y_train):
    """Create and train a simple classification model."""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model with a report."""
    from sklearn.metrics import classification_report, accuracy_score
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    data = import_data('data_with_pct_changes.csv')
    feature_columns = [col for col in data.columns if col not in ['countrycode', 'year', 'recession_next_year']]
    label_column = 'recession_next_year'
    
    X, y = split_features_labels(data, feature_columns, label_column)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = create_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Model Accuracy: {accuracy:.2f}')