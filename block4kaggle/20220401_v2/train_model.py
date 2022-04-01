def train_tree_model_for_ranker(
        data,
        selected_features,
        customer_id,
        label,
        mode='lgb',
        paras=None,
):
    data.sort_values(by=[customer_id], inplace=True)
    g_data = data.groupby([customer_id], as_index=False).count()[label].values
    if mode == 'lgb':
        if paras is None:
            paras = {
                'n_estimators': 1000,
                'boosting_type': 'gbdt',
            }
        ranker = lgb.LGBMRanker(**paras)

    elif mode == 'xgb':
        if paras is None:
            paras = {
                'objective': 'binary:logistic',
                'n_estimators': 1000,
            }
        clf = xgb.XGBClassifier(**paras)
    ranker.fit(data[selected_features], data[[label]], group=g_data)
    return ranker


'''
above is code
---------------------------------------------------------
below is running code
'''

clf = train_tree_model_for_ranker(
    data=samples_feature4finedrank_trn_cv,
    selected_features=selected_features,
    customer_id='customer_id',
    label='label',
    mode='lgb'
)