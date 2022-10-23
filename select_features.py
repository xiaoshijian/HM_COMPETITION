from .mvtest_stat import mvtest
from typing import List
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import chi2
from sklearn.preprocessing import minmax_scale

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
    
    
 class ContinousFeatureFilter:
    def __init__(self, colnames2filter, colname4id, colname4label, ratio=0.1):
        self._colname4id = colname4id
        self._colname4label = colname4label
        self._colnames2filter = [col for col in colnames2filter if col not in [self._colname4id, self._colname4label]] 
        self._ratio = ratio
        
    def fit(self, data):
        # 使用chi2来筛选生成的一系列连续型特征
        m = len(self._colnames2filter)
        chi2_values = chi2(
            # 这种replace的处理方法是不完善的，可以继续改进的
            minmax_scale(data[self._colnames2filter].replace([np.inf, -np.inf, np.nan], 0)),  # 填入的需要是non-negative的，所以进行minmax
            data[self._colname4label])[0]  
        pairs = [[col, value] for col, value in zip(self._colnames2filter, chi2_values.tolist())]
#         print(pairs)
        pairs = sorted(pairs, key = lambda x: x[1], reverse=True)
        # record some infomation why to do feature selection decison, 
        # features in self._features_chi2info should be more than features in self._selected_features
        self._features_chi2info = {x[0]: x[1] for x in pairs}  
        self._selected_features = [x[0] for x in pairs[:int(m * self._ratio)]]
    
    def transform(self, data):
        return data[[self._colname4id] + self._selected_features]
