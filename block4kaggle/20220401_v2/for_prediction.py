import time
import pandas as pd


def timer(func):
    def func_wrapper(*arg, **kwargs):
        t0 = time.time()
        print('begin to run function "{}"'.format(func.__name__))
        result = func(*arg, **kwargs)
        t1 = time.time()
        t_diff = t1 - t0
        msg = 'finish function "{}", time used: {}m, {}s'
        print(msg.format(func.__name__, t_diff // 60, t_diff % 60))
        return result

    return func_wrapper


@timer
def batch_pred(batch_samples,
               clf,
               article_feature_dict,
               customer_feature_dict,
               interaction_feature_dict,
               features_name,
               max_recommend_num=12):
    # generate data for prediction
    data4pred = []
    for cid, aids in zip(
            batch_samples['customer_id'], batch_samples['recall_items']
    ):
        for aid in aids:
            info = {}
            info['customer_id'] = cid
            info['article_id'] = aid
            features = combine_info_for_ranker(
                aid,
                cid,
                article_feature_dict,
                customer_feature_dict,
                interaction_feature_dict,
            )
            info.update(features)
            data4pred.append(info)
    data4pred = pd.DataFrame(data4pred)
    data4pred['score'] = clf.predict(data4pred[features_name], num_iteration=clf.best_iteration_)

    data4pred['aid_n_score'] = data4pred.apply(
        lambda x: (x.article_id, x.score), axis=1)
    # merge and combine records from same customer into list
    data4pred = data4pred.groupby('customer_id')['aid_n_score'].apply(list).reset_index(name='recommend')

    # sort items in recommen by score
    data4pred['recommend'] = data4pred['recommend'].apply(
        lambda x: sorted(x, key=lambda x: x[1], reverse=True)
    )
    # extract recommend article
    data4pred['recommend'] = data4pred['recommend'].apply(
        lambda x: [item[0] for item in x][:max_recommend_num]
    )
    return data4pred


def pred(samples,
         clf,
         article_feature_dict,
         customer_feature_dict,
         interaction_feature_dict,
         features_name,
         batch_size=1000,
         max_recommend_num=12):
    # tss is abbrevation for total sample size
    tss = len(samples)
    prediction = []
    for i in range(0, tss, batch_size):
        if i == tss - 1:
            continue
        bsp = i
        bep = min(bsp + batch_size, tss)
        batch_samples = samples[bsp:bep]
        prediction.append(
            batch_pred(
                batch_samples,
                clf,
                article_feature_dict,
                customer_feature_dict,
                interaction_feature_dict,
                features_name,
                max_recommend_num
            )
        )
        # 是否要加del啊？
    predicton = pd.concat(prediction, axis=0, ignore_index=True)
    return predicton


'''
above is code
---------------------------------------------------------
below is running code
'''

prediction_dev_cv = pred(
    recall_and_purchased_dev_cv,
    clf,
    arr_dev_cv,
    apk_dev_cv,
    hpi_dev_cv,
    selected_features,
)
