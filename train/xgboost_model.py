import xgboost as xgb

d_train = xgb.DMatrix('data/processed/train_set.buffer')
d_val = xgb.DMatrix('data/processed/val_set.buffer')

params = {
    'objective': 'multi:softmax',
    'num_class': 18,
    'eval_metric': 'mlogloss',  
    'eta': 0.1,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 69
}

watchlist = [(d_train, 'train'), (d_val, 'eval')]
num_boost_round = 250 

print("\nStarting model training...")

model = xgb.train(
    params=params,
    dtrain=d_train,
    num_boost_round=num_boost_round,
    evals=watchlist,
    early_stopping_rounds=15, 
    verbose_eval=1            
)

model.save_model('models/xgboost_model_v2.json')
