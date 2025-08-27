from scipy.stats import f
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib

train_data = pd.read_csv('cmi-detect-behavior-with-sensor-data/train.csv')
test_data = pd.read_csv('cmi-detect-behavior-with-sensor-data/test.csv')
train_demo = pd.read_csv('cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test_demo = pd.read_csv('cmi-detect-behavior-with-sensor-data/test_demographics.csv')
OUTPUT_PATH = 'data/processed'

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
                    'rot_z']

# --- Creating train/test splits to ensure all sequences stay in groups ---
X = imu_only_data.drop(columns=['gesture'])
y = imu_only_data['gesture']
groups = imu_only_data['sequence_id']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=69)
train_idx, val_idx = next(gss.split(X, y, groups))

train_df = imu_only_data.iloc[train_idx]
val_df = imu_only_data.iloc[val_idx]

print(f"df before encoding shape: {imu_only_data.shape[0]} x {imu_only_data.shape[1]}")


print(f"train_x: {train_df.shape[0]} x {train_df.shape[1]}")
print(f"val_x: {val_df.shape[0]} x {val_df.shape[1]}")

le = LabelEncoder()

train_X = csr_matrix(train_df[numerical_cols].values)
val_X = csr_matrix(val_df[numerical_cols].values)
train_y = le.fit_transform(train_df['gesture'])
val_y = le.transform(val_df['gesture'])

d_train = xgb.DMatrix(train_X, label=train_y)
d_val = xgb.DMatrix(val_X, label=val_y)


d_train.save_binary(f"{OUTPUT_PATH}/train_set.buffer") 
d_val.save_binary(f"{OUTPUT_PATH}/val_set.buffer")
joblib.dump(le, 'models/label_encoder.joblib')