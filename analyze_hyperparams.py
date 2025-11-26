import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress pandas warnings about future behavior
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration
OUTPUT_DIR = 'output'
IMG_DIR = 'analysis_images'
os.makedirs(IMG_DIR, exist_ok=True)

def get_accuracy(cm):
    """Calculate accuracy from confusion matrix list."""
    cm = np.array(cm)
    # Handle edge case where matrix might be empty or zero
    if cm.sum() == 0:
        return 0.0
    return np.trace(cm) / np.sum(cm)

def print_parameter_stats(df, param):
    """
    Prints detailed summary statistics for a specific hyperparameter.
    Handles both discrete values and continuous binning.
    """
    print(f"\n🔹 Breakdown by: {param}")
    
    # Work on a copy to avoid modifying the main loop's dataframe
    local_df = df.copy()
    
    is_numeric = pd.api.types.is_numeric_dtype(local_df[param])
    unique_count = local_df[param].nunique()
    group_col = param

    # If numeric and too many unique values (e.g. random float search), bin them
    if is_numeric and unique_count > 15:
        try:
            # Create 5 bins for readability
            bin_col_name = f"{param}_range"
            local_df[bin_col_name] = pd.cut(local_df[param], bins=5)
            group_col = bin_col_name
            print(f"(Continuous variable with {unique_count} unique values - grouped into 5 bins)")
        except Exception:
            pass # Fallback to exact grouping if binning fails

    # Group and Aggregate
    try:
        stats = local_df.groupby(group_col)['accuracy'].agg(
            Trials='count',
            Mean='mean',
            Max='max',
            Std='std'
        )
        
        # Fill NaN std dev (happens if only 1 trial exists for a value)
        stats['Std'] = stats['Std'].fillna(0.0)
        
        # Sort by Mean Accuracy descending (Best configurations on top)
        stats = stats.sort_values(by='Mean', ascending=False)
        
        # Print formatted table
        print(stats.to_markdown(floatfmt=".4f"))
        
    except Exception as e:
        print(f"Could not generate stats for {param}: {e}")

def analyze_model_trials(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    model_name = data['model_name']
    trials = data.get('trials', [])
    
    if not trials:
        print(f"Skipping {model_name}: No trial data found.")
        return

    print(f"\n{'='*60}")
    print(f" ANALYZING: {model_name} ({len(trials)} trials)")
    print(f"{'='*60}")

    # 1. Flatten Data
    records = []
    for trial in trials:
        row = trial['hyperparameters'].copy()
        row['accuracy'] = get_accuracy(trial['confusion_matrix'])
        records.append(row)
    
    df = pd.DataFrame(records)
    
    # --- NEW: Global Stats Calculation ---
    print("\n--- Global Statistics (All Trials) ---")
    # Calculate aggregate stats for the 'accuracy' column
    global_stats = df['accuracy'].agg(['count', 'mean', 'max', 'min', 'std'])
    
    # Convert to a DataFrame for nice markdown printing
    global_stats_df = global_stats.to_frame().T
    print(global_stats_df.to_markdown(floatfmt=".4f"))
    # -------------------------------------

    # 2. Identify Parameters (exclude accuracy)
    param_cols = [c for c in df.columns if c != 'accuracy']
    
    # 3. Generate Detailed Stats Table for EACH Parameter
    for param in param_cols:
        print_parameter_stats(df, param)

    # 4. Generate Plots
    try:
        # Calculate grid size for subplots
        n_params = len(param_cols)
        if n_params > 0:
            n_cols = 2
            n_rows = (n_params + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            fig.suptitle(f'Hyperparameter Sensitivity: {model_name}', fontsize=16)
            
            # Handle case where there is only 1 parameter (axes is not a list)
            if n_params == 1:
                axes = np.array([axes])
                
            axes = axes.flatten() 

            for i, param in enumerate(param_cols):
                ax = axes[i]
                
                # Check data type
                if pd.api.types.is_numeric_dtype(df[param]):
                    # Numeric: Scatter plot with trend line
                    sns.regplot(data=df, x=param, y='accuracy', ax=ax, 
                                scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
                    
                    # Check for log scale need (if max/min ratio is huge)
                    if df[param].min() > 0 and (df[param].max() / df[param].min() > 100):
                        ax.set_xscale('log')
                        ax.set_title(f'{param} (Log Scale) vs Accuracy')
                    else:
                        ax.set_title(f'{param} vs Accuracy')
                        
                else:
                    # Categorical/String: Box plot
                    sns.boxplot(data=df, x=param, y='accuracy', ax=ax, palette='viridis')
                    ax.set_title(f'{param} vs Accuracy')

                ax.set_ylabel('Accuracy')
                ax.set_xlabel(param)

            # Hide empty subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
            
            # Save plot
            safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            save_path = os.path.join(IMG_DIR, f'hyp_{safe_name}.png')
            plt.savefig(save_path)
            print(f"\n[Graph] Saved plot to {save_path}")
            plt.close()
    except Exception as e:
        print(f"Error generating plots: {e}")

    # 5. Global Correlation Summary (Numeric only)
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1: # needs at least accuracy + 1 param
        print("\n--- Global Correlation with Accuracy ---")
        correlations = numeric_df.corr()['accuracy'].drop('accuracy').sort_values(key=abs, ascending=False)
        print(correlations.to_markdown(floatfmt=".4f"))

def main():
    print(">>> STARTING DETAILED HYPERPARAMETER ANALYSIS <<<")
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Directory '{OUTPUT_DIR}' not found.")
        return

    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
    
    if not files:
        print("No JSON files found in output directory.")
        return

    for file in files:
        try:
            analyze_model_trials(os.path.join(OUTPUT_DIR, file))
        except Exception as e:
            print(f"Could not analyze {file}: {e}")

    print("\n>>> ANALYSIS COMPLETE <<<")
    print(f"Check '{IMG_DIR}' for the generated figures.")

if __name__ == "__main__":
    main()
