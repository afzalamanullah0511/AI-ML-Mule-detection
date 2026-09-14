import pandas as pd
import xgboost as xgb

print("Loading dataset...")
df = pd.read_csv('xgboost_ready_bank_data.csv')

# 1. Isolate the target variable
target_name = 'F3924' if 'F3924' in df.columns else '3924'
y = df[target_name]

# 2. Apply the strict behavioral filter
drop_cols = [col for col in df.columns if 'Unnamed' in col or 'F2230' in col]
for i in range(3900, 3924):
    col_str = f'F{i}'
    if col_str in df.columns:
        drop_cols.append(col_str)

X = df.drop(columns=drop_cols + [target_name], errors='ignore')

# 3. Calculate optimized class weights
imbalance_weight = (len(y) - sum(y)) / sum(y)

print(f"Training the FINAL robust model on {X.shape[1]} pure behavioral features...")

final_model = xgb.XGBClassifier(
    scale_pos_weight=imbalance_weight, 
    eval_metric='auc',
    random_state=42,
    n_estimators=120,       # Slightly more trees to absorb patterns fully
    
    # --- DEFENSIVE REGULARIZATION ---
    max_depth=4,            # Smoother decision boundaries to prevent overfitting
    min_child_weight=3,     # Requires rules to cover multiple accounts
    subsample=0.8,          # Row regularization
    colsample_bytree=0.8,   # Feature regularization
    
    # --- GPU ENGINE SETTINGS ---
    tree_method='hist',
    device='cuda'
)

# Train on 100% of the available data
final_model.fit(X, y)

# Save the official model brain
model_filename = 'final_pure_behavioral_mule_hunter.json'
final_model.save_model(model_filename)

print("\n=============================================================")
print(f" SUCCESS: Final AI model saved as '{model_filename}'")
print("=============================================================")
print("Your production-ready model is locked and optimized for the GPU.")