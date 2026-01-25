from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os
from datetime import datetime

from sklearn.metrics import auc, roc_curve

def get_file_locations(base_output_dir='Experiments', experiments=None, file_extension='.pkl', includes_name=""):

    file_paths = []

    if experiments is None:
        # Search all experiment directories
        for date_dir in os.listdir(base_output_dir):
            date_path = os.path.join(base_output_dir, date_dir)
            if os.path.isdir(date_path):
                for exp_num in os.listdir(date_path):
                    exp_path = os.path.join(date_path, exp_num)
                    if os.path.isdir(exp_path):
                        for file in os.listdir(exp_path):
                            if file.endswith(file_extension) and includes_name in file:
                                file_paths.append(os.path.join(exp_path, file))
    else:
        # Search specified experiments
        for date_str, exp_num in experiments:
            exp_path = os.path.join(base_output_dir, date_str, f'experiment_{exp_num}')
            if os.path.isdir(exp_path):
                for file in os.listdir(exp_path):
                    if file.endswith(file_extension) and includes_name in file:
                        file_paths.append(os.path.join(exp_path, file))

    return file_paths

def get_results_from_file(file_path):
    """Load model results from a JSON file."""
    return pd.read_json(file_path, typ='series').to_dict()

def compare_model_performance(model_results_list, 
                             metrics=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
                             model_names=None,
                             comparison_dir=None):
    """Compare performance metrics of multiple models.
    
    Args:
        model_results_list: List of dictionaries containing model results (will extract 'metrics' sub-dict)
        metrics: List of metric names to compare
        model_names: List of names for the models. If None, uses 'Model 1', 'Model 2', etc.
    
    Returns:
        DataFrame with comparison of all models
    """
    # Generate default model names if not provided
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(model_results_list))]
    
    # Ensure we have the right number of names
    if len(model_names) != len(model_results_list):
        raise ValueError("Number of model names must match number of model results")
    
    # Extract 'metrics' dictionary from each model result
    metrics_list = []
    for model_results in model_results_list:
        if isinstance(model_results, dict) and 'metrics' in model_results:
            metrics_list.append(model_results['metrics'])
        else:
            # If 'metrics' key doesn't exist, assume the dict itself contains metrics
            metrics_list.append(model_results)
    
    comparison_data = []
    
    for metric in metrics:
        row = {'Metric': metric}
        
        # Get values for each model
        values = []
        for i, model_metrics in enumerate(metrics_list):
            val = model_metrics.get(metric, None)
            row[model_names[i]] = val
            if val is not None:
                values.append((val, model_names[i]))
        
        # Determine best model for this metric (highest value)
        if values:
            best_value, best_model = max(values, key=lambda x: x[0])
            row['Best Model'] = best_model
            row['Best Value'] = best_value
        else:
            row['Best Model'] = None
            row['Best Value'] = None
        
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison DataFrame if directory provided
    if comparison_dir is not None:
        os.makedirs(comparison_dir, exist_ok=True)
        comparison_path = os.path.join(comparison_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Comparison saved to {comparison_path}")

    return comparison_df

def calculate_roc_from_predictions(predictions_df, positive_class='Recession'):
    """Calculate ROC curve from prediction probabilities.
    
    Args:
        predictions_df: DataFrame with columns 'true_label' and probability columns
        positive_class: Name of the positive class column (default: 'Recession')
    
    Returns:
        fpr, tpr, roc_auc
    """
    # Get the probability column for the positive class
    prob_column = f'prob_{positive_class}'
    
    if prob_column not in predictions_df.columns:
        raise ValueError(f"Column '{prob_column}' not found in predictions DataFrame")
    
    # Extract true labels and predicted probabilities
    y_true = predictions_df['true_label'].values
    y_prob = predictions_df[prob_column].values
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

def plot_multiple_roc_curves_from_files(json_files, predictions_files, model_names=None, 
                                        title=None, figsize=(10, 8),
                                        positive_class='Recession', comparison_dir=None):
    """Plot ROC curves for multiple models using separate JSON and predictions files.
    
    Args:
        json_files: List of paths to JSON files containing metrics
        predictions_files: List of paths to CSV files containing predictions
        model_names: List of names for the models. If None, uses 'Model 1', 'Model 2', etc.
        title: Title for the plot
        figsize: Tuple for figure size (width, height)
        save_path: Optional path to save the figure. If None, displays the plot.
        positive_class: Name of the positive class for ROC calculation
    
    Returns:
        matplotlib figure and axis objects
    """
    # Ensure we have matching numbers of files
    if len(json_files) != len(predictions_files):
        raise ValueError(f"Number of JSON files ({len(json_files)}) must match number of predictions files ({len(predictions_files)})")
    
    # Generate default model names if not provided
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(json_files))]
    
    # Ensure we have the right number of names
    if len(model_names) != len(json_files):
        raise ValueError("Number of model names must match number of files")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for different models
    colors = plt.cm.tab10(range(len(json_files)))
    
    # Plot ROC curve for each model
    for i, (json_file, pred_file, model_name) in enumerate(zip(json_files, predictions_files, model_names)):
        # Load metrics from JSON
        metrics = get_results_from_file(json_file)
        if 'metrics' in metrics:
            metrics = metrics['metrics']
        
        # Try to get FPR and TPR from JSON first
        fpr = metrics.get('fpr', None)
        tpr = metrics.get('tpr', None)
        roc_auc = metrics.get('roc_auc', None)
        
        # If FPR and TPR not in JSON, calculate from predictions file
        if fpr is None or tpr is None:
            try:
                # Load predictions
                predictions_df = pd.read_csv(pred_file)
                
                # Calculate ROC from predictions
                fpr, tpr, roc_auc = calculate_roc_from_predictions(predictions_df, positive_class)
                print(f"Calculated ROC from predictions file for {model_name}")
            except Exception as e:
                print(f"Error calculating ROC for {model_name}: {e}")
                continue
        
        # Plot the curve
        if fpr is not None and tpr is not None:
            # Calculate AUC if not provided
            if roc_auc is None:
                roc_auc = auc(fpr, tpr)
            
            # Add markers only for 3-point curves (from confusion matrix)
            marker = 'o' if len(fpr) == 3 else None
            
            ax.plot(fpr, tpr, color=colors[i], lw=2, 
                   label=f'{model_name} (AUC = {roc_auc:.3f})',
                   marker=marker, markersize=6)
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier (AUC = 0.500)')
    

    title_text = 'ROC Curves Comparison: ' + title if title else 'ROC Curves Comparison'

    # Configure plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show the plot
    if comparison_dir:
        os.makedirs(comparison_dir, exist_ok=True)
        save_path = os.path.join(comparison_dir, 'roc_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC comparison plot saved to {save_path}")
    else:
        plt.show()
    
    return fig, ax

def plot_bar_comparison(comparison_df, model_names, metric_column='Metric', figsize=(10, 6), comparison_dir=None, title=None):
    """Plot bar comparison of model metrics.
    
    Args:
        comparison_df: DataFrame with comparison of model metrics
        model_names: List of model names to plot
        metric_column: Name of the column containing metric names
        save_path: Optional path to save the figure. If None, displays the plot.
        figsize: Tuple for figure size (width, height)
        title: Title for the plot
    Returns:
        matplotlib figure and axis objects
    """
    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set bar width and positions
    bar_width = 0.15
    indices = range(len(comparison_df))
    
    # Plot bars for each model with text labels
    for i, model_name in enumerate(model_names):
        values = comparison_df[model_name].values
        bar_positions = [index + i * bar_width for index in indices]
        ax.bar(bar_positions, values, width=bar_width, label=model_name)
        # Add value labels on top of bars
        for x, y in zip(bar_positions, values):
            ax.text(x, y + 0.01, f'{y:.2f}', ha='center', va='bottom', fontsize=8)

    title_text = 'Model Performance Comparison: ' + title if title else 'Model Performance Comparison'

    # Configure plot
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xticks([index + bar_width * (len(model_names) - 1) / 2 for index in indices])
    ax.set_xticklabels(comparison_df[metric_column], rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save or show the plot
    if comparison_dir:
        os.makedirs(comparison_dir, exist_ok=True)
        save_path = os.path.join(comparison_dir, 'bar_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bar comparison plot saved to {save_path}")
    else:
        plt.show()

    return fig, ax

def plot_confusion_matrixes(json_files, labels=['No Recession', 'Recession'], model_names=None,
                           title=None, figsize=(12, 8), comparison_dir=None):
    """Plot confusion matrices for multiple models using separate JSON and predictions files.
    
    Args:
        json_files: List of paths to JSON files containing metrics
        labels: List of labels for the confusion matrix (default: ['No Recession', 'Recession'])
        model_names: List of names for the models. If None, uses 'Model 1', 'Model 2', etc.
        title: Title for the plot
        figsize: Tuple for figure size (width, height)
        comparison_dir: Optional directory to save the figure. If None, displays the plot.
    Returns:
        matplotlib figure and axis objects
    """   
    # Generate default model names if not provided
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(json_files))]
    # Ensure we have the right number of names
    if len(model_names) != len(json_files):
        raise ValueError("Number of model names must match number of files")
    # Create figure
    num_models = len(json_files)
    fig, axes = plt.subplots(1, num_models, figsize=figsize)
    # Plot confusion matrix for each model
    for i, (json_file, model_name) in enumerate(zip(json_files, model_names)):
        # Load metrics from JSON
        metrics = get_results_from_file(json_file)
        if 'confusion_matrix' not in metrics:
            print(f"Confusion matrix not found in {json_file}, skipping.")
            continue
        cm = metrics['confusion_matrix']
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        # Plot heatmap
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(model_name, fontsize=12)
        axes[i].set_xlabel('Predicted Label', fontsize=10)
        axes[i].set_ylabel('True Label', fontsize=10)
        
    title_text = 'Confusion Matrices Comparison: ' + title if title else 'Confusion Matrices Comparison'
    plt.suptitle(title_text, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save or show the plot
    if comparison_dir:
        os.makedirs(comparison_dir, exist_ok=True)
        save_path = os.path.join(comparison_dir, 'confusion_matrices.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices plot saved to {save_path}")
    else:
        plt.show()
    return fig, axes

def create_comparison_directory(base_output_dir='Comparisons'):
    """Create a comparison directory with current date and next available experiment number."""
    date_str = datetime.now().strftime('%Y-%m-%d')
    date_dir = os.path.join(base_output_dir, date_str)
    os.makedirs(date_dir, exist_ok=True)

    # Find next available experiment number
    existing_experiments = [d for d in os.listdir(date_dir) if d.startswith('comparison_')]
    existing_numbers = [int(d.split('_')[1]) for d in existing_experiments if d.split('_')[1].isdigit()]
    next_exp_num = max(existing_numbers, default=0) + 1

    comparison_dir = os.path.join(date_dir, f'comparison_{next_exp_num}')
    os.makedirs(comparison_dir, exist_ok=True)

    return comparison_dir

if __name__ == "__main__":
    # Example usage
    base_output_dir = 'Experiments'
    experiments = [['2026-01-25', 5], ['2026-01-25', 8], ['2026-01-25', 6], ['2026-01-25', 7], ['2026-01-25', 9]]
    model_names = ['Recession in 1 year', 'Recession in 2 years', 'Recession in 3 years', 'Recession in 5 years', 'Recession in 10 years']
    metrics = ['f1_score', 'roc_auc', 'avg_precision']
    all_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'avg_precision']
    name_comparison = 'Forecast horizon change: 1 vs 2 vs 3 vs 5 vs 10 years'


    json_files = get_file_locations(base_output_dir, experiments, file_extension='.json')
    results_files = get_file_locations(base_output_dir, experiments, file_extension='.csv', includes_name='predictions')

    comparison_dir = create_comparison_directory(base_output_dir='Comparisons')

    model_results = [get_results_from_file(f) for f in json_files]
    comparison_df = compare_model_performance(model_results, metrics=metrics, model_names=model_names, comparison_dir=comparison_dir)
    print(comparison_df)

    # Plot ROC curves
    plot_multiple_roc_curves_from_files(json_files, results_files, model_names=model_names, comparison_dir=comparison_dir, title=name_comparison)

    # Plot bar comparison
    plot_bar_comparison(comparison_df, model_names=model_names, comparison_dir=comparison_dir, title=name_comparison)

    # Plot confusion matrices
    plot_confusion_matrixes(json_files, model_names=model_names, comparison_dir=comparison_dir, title=name_comparison)

    # Save full comparison with all metrics
    full_comparison_df = compare_model_performance(model_results, metrics=all_metrics, model_names=model_names, comparison_dir=comparison_dir)

