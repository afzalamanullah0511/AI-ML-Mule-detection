import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

print("Loading data...")
df = pd.read_csv('DataSet.csv') # Ensure this matches your original file name

numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(exclude=['number']).columns

print("1. Imputing missing numerical values...")
num_imputer = SimpleImputer(strategy='median')
num_imputed_data = num_imputer.fit_transform(df[numeric_cols])
df_num = pd.DataFrame(num_imputed_data, columns=num_imputer.get_feature_names_out())

print("2. Safely handling text columns...")
if len(categorical_cols) > 0:
    low_cardinality_cols = [col for col in categorical_cols if df[col].nunique() < 20]
    
    if len(low_cardinality_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        cat_imputed_data = cat_imputer.fit_transform(df[low_cardinality_cols])
        df_cat = pd.DataFrame(cat_imputed_data, columns=cat_imputer.get_feature_names_out())
        
        # Translate text to 1s and 0s
        df_cat = pd.get_dummies(df_cat, drop_first=True)
        df_imputed = pd.concat([df_num, df_cat], axis=1)
    else:
        df_imputed = df_num
else:
    df_imputed = df_num

print("3. Dropping blank and constant columns...")
selector = VarianceThreshold(threshold=0.0)
final_df = pd.DataFrame(selector.fit_transform(df_imputed), 
                          columns=df_imputed.columns[selector.get_support()])

print(f"\n--- SUCCESS ---")
print(f"Final shape ready for XGBoost: {final_df.shape}")

# 4. Save the file with a BRAND NEW NAME
final_df.to_csv('xgboost_ready_bank_data.csv', index=False)
print("Saved to your folder as: 'xgboost_ready_bank_data.csv'")