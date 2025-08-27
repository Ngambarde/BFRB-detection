import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = xgb.Booster()
model.load_model('models/xgboost_model_v2.json')

le = joblib.load('models/label_encoder.joblib')
d_val = xgb.DMatrix('data/processed/val_set.buffer')

y_pred = model.predict(d_val)
y_true = d_val.get_label()

accuracy = accuracy_score(y_true, y_pred)
print("\n--- Overall Model Performance ---")
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

class_names = le.classes_
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Gesture')
plt.xlabel('Predicted Gesture')
plt.show()