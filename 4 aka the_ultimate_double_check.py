import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df = pd.read_csv('xgboost_ready_bank_data.csv')
cheat_columns = [col for col in df.columns if 'Unnamed' in col or 'F2230' in col or 'F3912' in col]
if cheat_columns: df = df.drop(columns=cheat_columns)

target_name = 'F3924' if 'F3924' in df.columns else '3924'
y = df[target_name]
X = df.drop(columns=[target_name])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
imbalance_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# We use 'eval_set' to watch the training score vs test score in real-time
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