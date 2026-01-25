import pandas as pd
import matplotlib.pyplot as plt

def show_visualization_on_status():
    # Read the main CSV file and the country development status CSV
    df = pd.read_csv('engineered_features_all_countries.csv')
    dev_status = pd.read_csv('all_countries.csv')

    # Merge the dataframes (assuming both have a 'countrycode' column)
    df_merged = df.merge(dev_status, on='countrycode', how='left')

    # Define the development categories in order
    dev_categories = ['Developed', 'Developing', 'Least Developed']

    # Create figure with subplots for each development status
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Flatten axes for easier iteration
    ax_flat = axes.flatten()

    # Color scheme
    colors = ['#2ecc71', '#e74c3c']

    # First pass: find the maximum count across all categories to set uniform y-axis
    max_count = 0
    all_split_counts = {}
    for dev_cat in dev_categories:
        df_subset = df_merged[df_merged['status'] == dev_cat]
        split_counts = df_subset['recession_in_1y'].value_counts()
        split_counts.index = split_counts.index.map({0: 'No Recession', 1: 'Recession'})
        all_split_counts[dev_cat] = split_counts
        if len(split_counts) > 0:
            max_count = max(max_count, split_counts.max())

    # Process each development status
    for idx, dev_cat in enumerate(dev_categories):
        split_counts = all_split_counts[dev_cat]
        
        # Bar chart (top row)
        split_counts.plot(kind='bar', ax=ax_flat[idx], color=colors)
        ax_flat[idx].set_title(f'{dev_cat} Countries', fontsize=12, fontweight='bold')
        ax_flat[idx].set_xlabel('')
        ax_flat[idx].set_ylabel('Count', fontsize=10)
        ax_flat[idx].tick_params(axis='x', rotation=0)
        ax_flat[idx].grid(axis='y', alpha=0.3)
        
        # Set uniform y-axis limits
        ax_flat[idx].set_ylim(0, max_count * 1.1)
        
        # Add count labels on bars
        for i, v in enumerate(split_counts.values):
            ax_flat[idx].text(i, v + max_count * 0.02, str(v), 
                            ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Pie chart (bottom row)
        ax_flat[idx + 3].pie(split_counts.values, labels=split_counts.index, 
                            autopct='%1.1f%%', startangle=90, colors=colors,
                            textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax_flat[idx + 3].set_title(f'{dev_cat} - Proportion', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig('plots/Data_split.png')

def show_visualization_for_multiple_features(feature_cols, output_name):
    df = pd.read_csv('engineered_features_all_countries.csv')

    fig, axes = plt.subplots(2, len(feature_cols), figsize=(5 * len(feature_cols), 10))
    ax_flat = axes.flatten()

    # Fixed order and colors
    label_order = ['No Recession', 'Recession']
    color_map = {'No Recession': '#2ecc71', 'Recession': '#e74c3c'}

    # First pass: get max count for uniform y-axis
    max_count = 0
    all_split_counts = {}

    for feature in feature_cols:
        split_counts = df[feature].value_counts()
        split_counts.index = split_counts.index.map({0: 'No Recession', 1: 'Recession'})

        # Force order (even if one class is missing)
        split_counts = split_counts.reindex(label_order, fill_value=0)

        all_split_counts[feature] = split_counts
        max_count = max(max_count, split_counts.max())

    # Plot each feature
    for idx, feature in enumerate(feature_cols):
        split_counts = all_split_counts[feature]

        colors = [color_map[label] for label in split_counts.index]

        # Bar chart (top row)
        split_counts.plot(kind='bar', ax=ax_flat[idx], color=colors)
        ax_flat[idx].set_title(feature, fontsize=12, fontweight='bold')
        ax_flat[idx].set_xlabel('')
        ax_flat[idx].set_ylabel('Count', fontsize=10)
        ax_flat[idx].tick_params(axis='x', rotation=0)
        ax_flat[idx].grid(axis='y', alpha=0.3)
        ax_flat[idx].set_ylim(0, max_count * 1.1)

        # Add value labels
        for i, v in enumerate(split_counts.values):
            ax_flat[idx].text(i, v + max_count * 0.02, str(v),
                              ha='center', va='bottom',
                              fontweight='bold', fontsize=9)

        # Pie chart (bottom row)
        ax_flat[idx + len(feature_cols)].pie(
            split_counts.values,
            labels=split_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        ax_flat[idx + len(feature_cols)].set_title(f'{feature} - Proportion',
                                                   fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_name)
    plt.close()


show_visualization_for_multiple_features(
    ['recession_in_1y', 'recession_in_2y', 'recession_in_3y', 'recession_in_5y', 'recession_in_10y'],
    'plots/Data_split_all_horizons.png'
)