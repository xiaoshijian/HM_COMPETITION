from .mvtest_stat import mvtest
import xgboost as xgb
import lightgbm as lgb


def get_mv_for_features(data, label):
    model = mvtest()
    output = {}
    for col in data.columns:
        if col == label:
            continue
        try:
            mv4feature = model.test_accelerate(data[col], data[label])
            output[col] = mv4feature
        except Exception as e:
            print(col)
            print(e)
            pass
    return output



def get_importance_from_tree(data, label, paras=None, mode='lgb'):
    # use the api "feature_importances_" in tree model (skilearn)
    # to get feature importance
    columns_in_x = [col for col in data.columns if col != label]
    x = data[columns_in_x]
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
                'objective': 'binary:logistic',
                'n_estimators': 1000,
            }
        clf = xgb.XGBClassifier(**paras)
    clf.fit(x, y)
    feature_importance = clf.feature_importances_
    feature_and_importance = [(col, value) for col, value in zip(columns_in_x, feature_importance)]
    return (clf, feature_importance, feature_and_importance)
