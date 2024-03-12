import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import ndcg_score
from catboost import CatBoostClassifier, Pool

def train_test (model, train_Pool, test_Pool, X_test, y_test):
    model.fit(train_Pool, eval_set=test_Pool)
    predictions = model.predict_proba(X_test)[:, 1]
    result_score = ndcg_score(y_test.values.reshape(1, -1), predictions.reshape(1, -1), k=X_test.shape[0])
    print("NDCG_SCORE = ", result_score)
    return result_score

train_data = pd.read_csv("C:/Users/smyat/Desktop/Internship VK/train_df.csv")

test_data = pd.read_csv("C:/Users/smyat/Desktop/Internship VK/test_df.csv")

y_train = train_data['target']
X_train = train_data.drop(columns =['target'])

y_test = test_data['target']
X_test = test_data.drop(columns =['target'])

train_id = X_train['search_id']
test_id = X_test['search_id']

train_Pool = Pool(data=X_train, label=y_train, group_id=train_id)
test_Pool = Pool(data=X_test, label=y_test, group_id=test_id)

model = CatBoostClassifier(iterations=80, depth=6, random_strength=1.01, scale_pos_weight=0.78, custom_metric='NDCG', boosting_type = 'Plain', verbose = 10, objective = 'Logloss')
 
score = train_test(model, train_Pool, test_Pool, X_test, y_test)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

