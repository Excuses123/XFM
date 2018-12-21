import pandas as pd
import numpy as np
import lightgbm as lgb
from pandas import concat
from sklearn.preprocessing import LabelEncoder


def LGBTrain(data, category_features, numeric_features, label):
    lenTrain = len(data[0])
    data = concat(data)
    for n in category_features:
        data[n] = LabelEncoder().fit_transform(data[n])

    cols = category_features + numeric_features
    train = data[:lenTrain]
    test = data[lenTrain:]

    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 32,
        'learning_rate': 0.005,

        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'bagging_seed': 55,
        'seed': 77,

        'nthread': -1,
        'max_depth': -1,
        'verbose': 0
    }

    lgb_train = lgb.Dataset(train[cols], train[label].astype(int))
    lgb_test = lgb.Dataset(test[cols], test[label].astype(int))

    lgb_model = lgb.train(lgb_params, lgb_train,
                          num_boost_round=10000,
                          valid_sets=[lgb_train, lgb_test],
                          early_stopping_rounds=100,
                          verbose_eval=10)

    lgb_importance = pd.DataFrame({'gain': list(lgb_model.feature_importance(importance_type='gain')), 'feature': cols})
    print(lgb_importance.sort_values(by=['gain'], ascending=False).head(20))

    pred = lgb_model.predict(test[cols])

    return pred


