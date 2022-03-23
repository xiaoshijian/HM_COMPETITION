import xgboost as xgb
import lightgbm as lgb



def train_tree_model(data,
                     selected_features,
                     label,
                     mode='lgb',
                     paras=None):
    x = data[selected_features]
    y = data[[label]]
    if mode == 'lgb':
        if paras is None:
            paras = {
                'n_estimators': 1000,
            }
        clf = lgb.LGBMClassifier(**paras)

    elif mode == 'xgb':
        if paras is None:
            paras = {
                'objective' :'binary:logistic',
                'n_estimators': 1000,
            }
        clf = xgb.XGBClassifier(**paras)
    clf.fit(x, y)
    return clf
