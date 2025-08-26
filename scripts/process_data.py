from scipy.stats import f
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
import pandas as pd
import numpy as np

train_data = pd.read_csv('cmi-detect-behavior-with-sensor-data/train.csv')
test_data = pd.read_csv('cmi-detect-behavior-with-sensor-data/test.csv')
train_demo = pd.read_csv('cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test_demo = pd.read_csv('cmi-detect-behavior-with-sensor-data/test_demographics.csv')
OUTPUT_PATH = 'data/processed'

encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')


imu_only_data = train_data[['row_id', 
                            'sequence_type', 
                            'sequence_id', 
                            'sequence_counter', 
                            'subject', 
                            'phase', 
                            'gesture', 
                            'acc_x', 
                            'acc_y', 
                            'acc_z', 
                            'rot_w',
                            'rot_x',
                            'rot_y',
                            'rot_z',]]

numerical_cols = ['sequence_counter', 
                    'acc_x', 
                    'acc_y', 
                    'acc_z', 
                    'rot_w',
                    'rot_x',
                    'rot_y',
                    'rot_z',]

cat_cols = ['sequence_type', 
            'sequence_id', 
            'subject', 
            'phase']

# --- Creating train/test splits to ensure all sequences stay in groups ---
X = imu_only_data.drop(columns=['gesture'])
y = imu_only_data['gesture']
groups = imu_only_data['sequence_id']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=69)
train_idx, val_idx = next(gss.split(X, y, groups))

train_df = imu_only_data.iloc[train_idx]
val_df = imu_only_data.iloc[val_idx]

encoder.fit(train_df[cat_cols])
processed_df_list = []

print(f"df before encoding shape: {imu_only_data.shape[0]} x {imu_only_data.shape[1]}")


train_cat_encoded = encoder.transform(train_df[cat_cols])
train_num_encoded = csr_matrix(train_df[numerical_cols].values)

val_cat_encoded = encoder.transform(val_df[cat_cols])
val_num_encoded = csr_matrix(val_df[numerical_cols].values)

train_X = hstack([train_cat_encoded, train_num_encoded])
val_X = hstack([val_cat_encoded, val_num_encoded])

print(f"train_x: {train_X.shape[0]} x {train_X.shape[1]}")
print(f"val_x: {val_X.shape[0]} x {val_X.shape[1]}")

le = LabelEncoder()

train_y = le.fit_transform(train_df['gesture'])
val_y = le.transform(val_df['gesture'])

d_train = xgb.DMatrix(train_X, label=train_y)
d_val = xgb.DMatrix(val_X, label=val_y)


d_train.save_binary(f"{OUTPUT_PATH}/train_set.buffer") 
d_val.save_binary(f"{OUTPUT_PATH}/val_set.buffer") 