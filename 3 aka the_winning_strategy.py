import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("Loading dataset...")
df = pd.read_csv('xgboost_ready_bank_data.csv')

# 1. Isolate target variable
target_name = 'F3924' if 'F3924' in df.columns else '3924'
y = df[target_name]

# 2. STRICT FILTER: Drop ALL administrative/leak suspect columns
# We drop Unnamed, the F2230 dates, and EVERY feature from F3900 to F3923
drop_cols = [col for col in df.columns if 'Unnamed' in col or 'F2230' in col]

# Add all columns in the 3900 block to prevent database artifact leaks
for i in range(3900, 3924):
    col_str = f'F{i}'
    if col_str in df.columns:
        drop_cols.append(col_str)
        
print(f"Dropping {len(drop_cols)} potential system/metadata leak columns...")
X = df.drop(columns=drop_cols + [target_name], errors='ignore')

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
imbalance_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# 4. Train the Behavioral AI
print("\nTraining Pure Behavioral AI... (Zero metadata allowed)")
model = xgb.XGBClassifier(
    scale_pos_weight=imbalance_weight, 
    eval_metric='auc',
    random_state=42,
    n_estimators=100,
    max_depth=5,
    tree_method='hist',
    device='cuda'
)

model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=20)

print("\n--- PURE BEHAVIORAL PERFORMANCE REPORT ---")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

print("\n--- TRUE TRANSACTIONAL TOP 15 FEATURES ---")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(15))