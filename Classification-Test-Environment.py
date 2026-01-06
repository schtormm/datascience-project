import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor

import json
from datetime import datetime
import os

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

def split_data_random(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def split_data_time_based(X, y, split_year, year_col='year'):
    """Split the dataset into training and testing sets based on a year threshold."""
    train_indices = X[year_col] < split_year
    test_indices = X[year_col] >= split_year
    
    X_train = X[train_indices].drop(columns=[year_col])
    y_train = y[train_indices]
    
    X_test = X[test_indices].drop(columns=[year_col])
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def create_model(X_train, y_train, model=RandomForestClassifier(class_weight='balanced', random_state=42)):
    """Create and train a simple classification model."""
    model.fit(X_train, y_train)
    return model

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_classes)
    is_binary = n_classes == 2
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred)
    }
    
    if is_binary:
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1_score'] = f1_score(y_true, y_pred)
    else:
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    
    if y_pred_proba is not None:
        if is_binary:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba[:, 1])
        else:
            metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, 
                                                   multi_class='ovr', average='weighted')
            metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_pred_proba, 
                                                   multi_class='ovo', average='weighted')
    
    return metrics, is_binary


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, output_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def plot_roc_curve(y_true, y_pred_proba, roc_auc, model_name, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(y_true, y_pred_proba, avg_precision, model_name, output_path):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_classification_report(y_true, y_pred, class_names, model_name, timestamp, output_path):
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    
    with open(output_path, 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("="*60 + "\n\n")
        f.write(report_text)
    
    return report_dict


def save_metrics_json(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)


def print_evaluation_summary(results, is_binary, y_pred_proba, output_dir, file_paths):
    print(f"\n{'='*60}")
    print(f"Evaluation Results - {results['model_name']}")
    print(f"{'='*60}")
    print(f"\nAccuracy: {results['metrics']['accuracy']:.4f}")
    
    if is_binary:
        print(f"Precision: {results['metrics']['precision']:.4f}")
        print(f"Recall: {results['metrics']['recall']:.4f}")
        print(f"F1-Score: {results['metrics']['f1_score']:.4f}")
        if y_pred_proba is not None:
            print(f"ROC-AUC: {results['metrics']['roc_auc']:.4f}")
    else:
        print(f"Precision (Macro): {results['metrics']['precision_macro']:.4f}")
        print(f"Recall (Macro): {results['metrics']['recall_macro']:.4f}")
        print(f"F1-Score (Macro): {results['metrics']['f1_macro']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Confusion Matrix: {file_paths['confusion_matrix']}")
    print(f"  - Classification Report: {file_paths['classification_report']}")
    print(f"  - Metrics JSON: {file_paths['metrics_json']}")
    if 'roc_curve' in file_paths:
        print(f"  - ROC Curve: {file_paths['roc_curve']}")
        print(f"  - PR Curve: {file_paths['pr_curve']}")
    print(f"{'='*60}\n")


def get_next_experiment_number(base_dir):
    if not os.path.exists(base_dir):
        return 1
    
    existing_experiments = [d for d in os.listdir(base_dir) 
                           if os.path.isdir(os.path.join(base_dir, d)) 
                           and d.startswith('experiment_')]
    
    if not existing_experiments:
        return 1
    
    experiment_numbers = []
    for exp in existing_experiments:
        try:
            num = int(exp.split('_')[1])
            experiment_numbers.append(num)
        except (IndexError, ValueError):
            continue
    
    return max(experiment_numbers) + 1 if experiment_numbers else 1


def evaluate_classification_model(y_true, y_pred, y_pred_proba=None, 
                                   class_names=None, base_output_dir='Experiments',
                                   model_name='classifier'):
    # Setup directory structure: Experiments/{Date}/experiment_N
    date_str = datetime.now().strftime('%Y-%m-%d')
    date_dir = os.path.join(base_output_dir, date_str)
    
    # Get next experiment number
    exp_num = get_next_experiment_number(date_dir)
    output_dir = os.path.join(date_dir, f'experiment_{exp_num}')
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Determine class names
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_classes]
    
    # Initialize results
    results = {
        'model_name': model_name,
        'timestamp': timestamp,
        'experiment_number': exp_num,
        'output_directory': output_dir,
        'metrics': {}
    }
    
    # Calculate metrics
    metrics, is_binary = calculate_metrics(y_true, y_pred, y_pred_proba)
    results['metrics'] = metrics
    
    # Define file paths
    file_paths = {
        'confusion_matrix': os.path.join(output_dir, f'{model_name}_confusion_matrix_{timestamp}.png'),
        'classification_report': os.path.join(output_dir, f'{model_name}_classification_report_{timestamp}.txt'),
        'metrics_json': os.path.join(output_dir, f'{model_name}_metrics_{timestamp}.json')
    }
    
    # Generate confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred, class_names, model_name, file_paths['confusion_matrix'])
    results['confusion_matrix'] = cm.tolist()
    
    # Save classification report
    report_dict = save_classification_report(y_true, y_pred, class_names, model_name, 
                                            timestamp, file_paths['classification_report'])
    results['classification_report'] = report_dict
    
    # Generate ROC and PR curves for binary classification
    if y_pred_proba is not None and is_binary:
        file_paths['roc_curve'] = os.path.join(output_dir, f'{model_name}_roc_curve_{timestamp}.png')
        file_paths['pr_curve'] = os.path.join(output_dir, f'{model_name}_pr_curve_{timestamp}.png')
        
        plot_roc_curve(y_true, y_pred_proba, results['metrics']['roc_auc'], 
                      model_name, file_paths['roc_curve'])
        plot_precision_recall_curve(y_true, y_pred_proba, results['metrics']['avg_precision'], 
                                   model_name, file_paths['pr_curve'])
    
    # Save metrics to JSON
    save_metrics_json(results, file_paths['metrics_json'])
    
    # Print summary
    print_evaluation_summary(results, is_binary, y_pred_proba, output_dir, file_paths)
    
    print(f"Experiment saved as: {output_dir}")
    
    return results

def tune_xgboost(X_train, y_train, task='classification', search_method='grid', 
                 param_grid=None, n_iter=50, cv=5, scoring=None, n_jobs=-1, 
                 verbose=1, random_state=42):
    """
    Hyperparameter tuning for XGBoost models.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    task : str, default='classification'
        Either 'classification' or 'regression'
    search_method : str, default='grid'
        Either 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
    param_grid : dict, optional
        Custom parameter grid. If None, uses default comprehensive grid
    n_iter : int, default=50
        Number of iterations for RandomizedSearchCV
    cv : int, default=5
        Number of cross-validation folds
    scoring : str or callable, optional
        Scoring metric. If None, uses default for task
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all processors)
    verbose : int, default=1
        Verbosity level
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    dict : Contains 'best_model', 'best_params', 'best_score', 'cv_results'
    """
    
    # Set up base model
    if task == 'classification':
        from collections import Counter
        class_counts = Counter(y_train)
        
        if len(class_counts) == 2:
            neg, pos = class_counts[0], class_counts[1]
            scale_pos_weight = neg / pos
        else:
            scale_pos_weight = None

        base_model = XGBClassifier(random_state=random_state, eval_metric='aucpr', scale_pos_weight=min(scale_pos_weight, 2.5), n_estimators=1500)
        if scoring is None:
            scoring = 'average_precision'
    elif task == 'regression':
        base_model = XGBRegressor(random_state=random_state, eval_metric='rmse')
        if scoring is None:
            scoring = 'neg_mean_squared_error'
    else:
        raise ValueError("task must be 'classification' or 'regression'")
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.3],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }
    
    tscv = TimeSeriesSplit(n_splits=5)
    # Choose search method
    if search_method == 'grid':
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
    elif search_method == 'random':
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=tscv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            return_train_score=True
        )
    else:
        raise ValueError("search_method must be 'grid' or 'random'")
    
    # Fit the search
    print(f"Starting {search_method} search with {cv}-fold cross-validation...")
    search.fit(X_train, y_train)
    
    # Return results
    results = {
        'best_model': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': search.cv_results_
    }
    
    print(f"\nBest {scoring} score: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    
    return results

if __name__ == "__main__":
    data = import_data('engineered_features_all_countries.csv')
    feature_columns = [col for col in data.columns if col not in ['countrycode', 'recession_next_year']]
    label_column = 'recession_next_year'
    
    X, y = split_features_labels(data, feature_columns, label_column)
    # X_train, X_test, y_train, y_test = split_data_random(X, y)
    X_train, X_test, y_train, y_test = split_data_time_based(X, y, split_year=2000, year_col='year')

    results = tune_xgboost(
        X_train, y_train, 
        task='classification',
        search_method='random',
        n_iter=10,
        scoring=average_precision_score
    )

    best_params = results['best_params']

    final_model = XGBClassifier(
        random_state=42,
        eval_metric='aucpr',
        n_estimators=3000,
        **best_params
    )

    final_model.fit(
        X_train,
        y_train
    )

    test_score = final_model.score(X_test, y_test)
    print(f"\nTest set accuracy: {test_score:.4f}")

    # Get predictions
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)

    # # Evaluate the model
    results = evaluate_classification_model(
        y_true=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        class_names=['No Recession', 'Recession'],
        base_output_dir='experiments',
        model_name='XGBoost_example'
    )