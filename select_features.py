from .mvtest_stat import mvtest
from typing import List
import numpy as np
import pandas as pd
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

def select_features_by_fi(selected_features, feature_importances_list, ratio: float=0.8):
    # fi is abbrevation for features importance
    n = len(selected_features)
    amout2select = int(n * ratio)
    items = [(f, fi) for f, fi in zip(selected_features, feature_importances_list)]
    items = sorted(items, key=lambda x: x[1], reverse=True)
    select_features_and_fi = items[:amout2select]
    select_features = [item[0] for item in select_features_and_fi]
    return select_features, select_features_and_fi

def extract_features_from_three_importance_type(
        data: pd.DataFrame,
        selected_features: List[str],  
        label: str,
        ratio: float = 0.8,
        paras: Dict = None,
        boost_type='xgb'
    ):
        """
        1. 获取以不同的feature_importance方法得到的特征重要性，
        2. 并且获取那些同时存在于以不同的feature_importance方法得到的特征（就是比较robust的）
        """
        x = data[selected_features]
        y = data[[label]]
        if boost_type='xgb':
            result = {}
            occurrences_features = []
            for criteria in ["gain" , "weight" , "cover"]:
                if paras is None:
                    paras = {
                        'n_estimators': 1000,
                    }
                paras['importance_type'] = criteria
                clf = xgb.XGBClassifier(**paras)
                clf.fit(x, y)
                select_features, select_features_and_fi = select_features_by_fi(
                    selected_features,
                    clf.feature_importances_,
                    ratio
                )
                result[criteria] = selected_features
                occurrences_features.extend(select_features)
            result['occurrences_features'] = list(set(occurrences_features))                
        elif boost_type='lgb':
            result = {}
            occurrences_features = []
            for criteria in ["split","gain"]:
                if paras is None:
                    paras = {
                        'n_estimators': 1000,
                    }
                paras['importance_type'] = criteria
                clf = xgb.XGBClassifier(**paras)
                clf.fit(x, y)
                select_features, select_features_and_fi = select_features_by_fi(
                    selected_features,
                    clf.feature_importances_,
                    ratio
                )
                result[criteria] = selected_features
                occurrences_features.extend(select_features)
            result['occurrences_features'] = list(set(occurrences_features))
        return result 
