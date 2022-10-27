from typing import List
import lightgbm as lgb
import category_encoders as ce
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class FindBestEncoder4Cat:
    """
    通过两次逐步迭代的方法，找出对类别型特征最好的编码方式
    """
    def __init__(
            self,
            colnames_for_categorical_columns: List,
            colname4id,
            colname4label,
            clf_name: str = 'lgb',
            ratio: float = 0.2, # ratio for testing set in whole data for selecting best encoder
            eval_func = roc_auc_score # it is eval function
            
    ):
        self._colnames_for_categorical_columns = colnames_for_categorical_columns
        self._colname4id = colname4id
        self._colname4label = colname4label
        self._clf_name = clf_name
        self._ratio = ratio
        self._eval_func = eval_func
    
    def _fit(self, data):
        # baseline for encoding categorical column is OrdinalEncoder
        tmp = []
        self._col_and_best_encoder_mapping = {}
        for i in range(len(self._colnames_for_categorical_columns)):
            colname2test = self._colnames_for_categorical_columns[i] # column to test which encoder is best for it
            rest_cat_colnames = [col for col in self._colnames_for_categorical_columns if col != colname2test]
            best_encoder_name = self._find_best_encoder(data, colname2test, rest_cat_colnames)
            tmp.append([colname2test, best_encoder_name])
        for colname, best_encoder_name in tmp:
            best_encoder = self._fit_best_encoder(data, colname, best_encoder_name)
            self._col_and_best_encoder_mapping[colname] = best_encoder
        return    
    
    def fit(self, data):
        return self._fit(data)
    
    def _transform(self, data):
        for colname, best_encoder in self.col_and_best_encoder_mapping.items():
            data = best_encoder.transform(data)
        return data

    def transform(self, data):
        return self._transform(data)
    
    def _find_best_encoder(self, data, colname2test, rest_cat_colnames):
        print('----------begin to find best encoder for {}----------'.format(colname2test))
        best_score, best_encoder_name = 0, None
        # baseline for encoding categorical column is OrdinalEncoder
        encoder2test = ['ordinal', 'one-hot', 'target', 'woe', 'hashing', 'count']  # 还可以加其它的
        encoder4rest = ce.OrdinalEncoder(rest_cat_colnames)
        data = encoder4rest.fit_transform(data)
        train_data, test_data = train_test_split(data, test_size=self._ratio, random_state=1992)
        for encoder_name in encoder2test:
            train_data_cpy = deepcopy(train_data)
            test_data_cpy = deepcopy(test_data)
            if encoder_name == 'ordinal':
                encoder = ce.ordinal.OrdinalEncoder([colname2test])
            elif encoder_name == 'one-hot':
                encoder = ce.one_hot.OneHotEncoder([colname2test])
            elif encoder_name == 'target':
                encoder = ce.target_encoder.TargetEncoder([colname2test])
            elif encoder_name == 'woe':
                encoder = ce.woe.WOEEncoder([colname2test])
            elif encoder_name == 'hashing':
                encoder = ce.hashing.HashingEncoder([colname2test])
            elif encoder_name == 'count':
                encoder = ce.count.CountEncoder([colname2test])
            train_data_cpy = encoder.fit_transform(
                    train_data_cpy, # 好像是不需要除去colname4id和colname4label的
                    train_data_cpy[self._colname4label]
                )
            test_data_cpy = encoder.transform(test_data_cpy)
            score = self._check_encoder_effect(train_data_cpy, test_data_cpy)
            if score > best_score:  # 暂时只考虑越大越好的情况
                best_encoder_name = encoder_name
                best_score = score
                msg = 'best encoder is UPDATED, current best encoder for {} is {}, and its score is {}'
                print(msg.format(colname2test, encoder_name, best_score))

        print('----------best encoder for {} is {}, best_score is ----------'.format(colname2test, best_encoder_name, best_score))
        return best_encoder_name

    def _fit_best_encoder(self, data, colname, best_encoder_name):
        # 对单个column训练encoder
        if best_encoder_name == 'ordinal':
            best_encoder = ce.ordinal.OrdinalEncoder([colname])
        elif best_encoder_name == 'one-hot':
            best_encoder = ce.one_hot.OneHotEncoder([colname])
        elif best_encoder_name == 'target':
            best_encoder = ce.target_encoder.TargetEncoder([colname])
        elif best_encoder_name == 'woe':
            best_encoder = ce.woe.WOEEncoder([colname])
        elif best_encoder_name == 'hashing':
            best_encoder = ce.hashing.HashingEncoder([colname])
        elif best_encoder_name == 'count':
            best_encoder = ce.count.CountEncoder([colname])
        best_encoder.fit(data, data[self._colname4label])
        return best_encoder

    def _check_encoder_effect(self, train_data, test_data):
        if self._clf_name == 'lgb':
            #TODO(XIAOSHIJIAN), 可能考虑加入输入paras的方法 
            clf = lgb.LGBMClassifier(paras = {
                'n_estimators': 1000,
            })
        clf.fit(
            train_data[
                [col for col in train_data.columns if col not in [self._colname4id, self._colname4label]]
                ],
            train_data[[self._colname4label]]
        )
        test_pred = clf.predict_proba(
            test_data[
                [col for col in test_data.columns if col not in [self._colname4id, self._colname4label]]
                ]
        )[:, 1]
        score = self._eval_func(test_data[[self._colname4label]], test_pred)
        return score
