import pandas as pd

print("Loading dataset for behavioral profiling...")
df = pd.read_csv('xgboost_ready_bank_data.csv')

# Find the exact name of the target column
target_name = 'F3924' if 'F3924' in df.columns else '3924'

# The top 5 features your AI discovered
top_features = ['F3914', 'F3898', 'F3907', 'F3920', 'F1639']

print("\n=============================================")
print("      INTERROGATING THE AI'S TOP SUSPECTS    ")
print("=============================================\n")

for feature in top_features:
    if feature in df.columns:
        print(f"--- Profiling Feature: {feature} ---")
        
        # Calculate statistics for Normal Accounts (0) vs Mule Accounts (1)
        stats = df.groupby(target_name)[feature].agg(['mean', 'min', 'max', 'std'])
        
        print(f"Normal Customers (0.0):")
        print(f"  Average: {stats.loc[0.0, 'mean']:.4f} | Min: {stats.loc[0.0, 'min']:.4f} | Max: {stats.loc[0.0, 'max']:.4f}")
        
        print(f"Mule Accounts (1.0):")
        print(f"  Average: {stats.loc[1.0, 'mean']:.4f} | Min: {stats.loc[1.0, 'min']:.4f} | Max: {stats.loc[1.0, 'max']:.4f}")
        print("-" * 45 + "\n")
    else:
        print(f"Feature {feature} not found in dataset.\n")