from copy import deepcopy
import pandas as pd


def extract_arr_features(article, transaction, na_filler=-1):
    '''
    # amount, ratio, rank in days [0, 1, 2, 3, 4, 5, 6] for product item
    # amount, ratio, rank in tw, lw, llw, lllw for product item

    :param article: dataframe that record article infor
    :param transaction: record for transactions
    :param na_filler: value to fill na
    :return:
    '''

    # arr is abbrevation for amount, ratio, rank
    output = deepcopy(article['article_id'])
    days = [i for i in range(7)]  #
    lds = [i for i in range(4)]  # 一周的最后一天，一个数字(code)就是代表的一周
    lds_feat_names = ['tw', 'lw', 'llw', 'lllw']
    min_dcode = transaction['dcode'].min()  # get mininal dcode in transaction record, which is the most recent days
    min_ldcode = transaction['ldcode'].min()  # get mininal ldcode in transaction record which in the most recent ld

    # generate feature with amount in days
    for day in days:
        cond = (transaction['dcode'] - min_dcode == day)
        feat_name = 'amount_day{}'.format(day)
        subsample = transaction[cond].reset_index(drop=True)  # extract transaction satisfy condition
        extracted_feat = subsample.groupby(['article_id'])['article_id'].count().reset_index(name=feat_name)
        output = pd.merge(output,
                          extracted_feat,
                          how='left',
                          on='article_id')

    # generate feature with amount in ld
    for ld, feat_name in zip(lds, lds_feat_names):
        cond = (transaction['ldcode'] - min_ldcode == ld)
        feat_name = 'amount_{}'.format(feat_name)
        subsample = transaction[cond].reset_index(drop=True)  # extract transaction satisfy condition
        extracted_feat = subsample.groupby(['article_id'])['article_id'].count().reset_index(name=feat_name)
        output = pd.merge(output,
                          extracted_feat,
                          how='left',
                          on='article_id')

    # add temperal feature total amount
    total_amount = transaction.groupby(['article_id'])['article_id'].count().reset_index(name='total_amount')
    output = pd.merge(output,
                      total_amount,
                      how='left',
                      on='article_id')
    for col in output.columns:
        if (col == 'article_id') or (col == 'total_amount'):
            continue
        _, timespan = col.split('_')
        feat_name_rank = 'rank_{}'.format(timespan)
        feat_name_ratio = 'ratio_{}'.format(timespan)
        output[feat_name_ratio] = output[col] / output['total_amount']  # add ratio feature
        output[feat_name_rank] = output[col].argsort()  # add rank feature
    output = output.drop(columns=['total_amount'])
    output.fillna(na_filler, inplace=True)
    return output


from copy import deepcopy
import pandas as pd

'''
for customer
'''
# amount, price, kingds of item customer buying in tw, lw, llw, lllw
def extract_apk_features(customer, transaction, na_filler=-1):
    '''
    :param article: dataframe that record customer infor
    :param transaction: record for transactions
    :param na_filler: value to fill na
    :return:
    '''
    output = deepcopy(customer)
    # generate feature with amount in ld
    min_ldcode = transaction['ldcode'].min()  # get mininal ldcode in transaction record
    lds = [i for i in range(4)]  # 一周的最后一天，一个数字(code)就是代表的一周
    lds_feat_names = ['tw', 'lw', 'llw', 'lllw']

    for ld, feat_name in zip(lds, lds_feat_names):
        cond = (transaction['ldcode'] - min_ldcode == ld)
        feat_name = 'amount_{}'.format(feat_name)
        subsample = transaction[cond].reset_index(drop=True)  # extract transaction satisfy condition

        amount_feat = 'c_amount_{}'.format(feat_name)
        price_feat = 'c_price_{}'.format(feat_name)
        kind_feat = 'c_kind_{}'.format(feat_name)

        # extract amount
        customer_amount = subsample.groupby(['customer_id'])['article_id'].count().reset_index(name=amount_feat)
        output = pd.merge(output, customer_amount, how='left', on='customer_id')

        # extract price
        extracted_price = subsample.groupby(['customer_id'])['price'].sum().reset_index(name=price_feat)
        output = pd.merge(output, extracted_price, how='left', on='customer_id')

        extracted_kind = subsample.groupby(['customer_id'])['article_id'].apply(set).apply(len).reset_index(name=kind_feat)
        output = pd.merge(output, extracted_kind, how='left', on='customer_id')

    output.fillna(na_filler, inplace=True)
    return output



'''
mainly useing for get interacting features
'''
def extract_historical_purchased_items(transaction):
    min_ldcode = transaction['ldcode'].min()
    lds_feat_names = ['tw', 'lw', 'llw', 'lllw']

    infor = {}
    for cid, aid, ldcode in zip(
        transaction['customer_id'],
        transaction['article_id'],
        transaction['ldcode']
    ):
        if cid not in infor:
            infor[cid] = {}
        if aid not in infor[cid]:
            infor[cid][aid] = {}
        if ldcode - min_ldcode == 0:
            if lds_feat_names[0] not in infor[cid][aid]: # this week
                infor[cid][aid][lds_feat_names[0]] = 0
            infor[cid][aid][lds_feat_names[0]] += 1
        elif ldcode - min_ldcode == 1:  # last week
            if lds_feat_names[1] not in infor[cid][aid]:
                infor[cid][aid][lds_feat_names[1]] = 0
            infor[cid][aid][lds_feat_names[1]] += 1
        elif ldcode - min_ldcode == 2:  # past of last week
            if lds_feat_names[2] not in infor[cid][aid]:
                infor[cid][aid][lds_feat_names[2]] = 0
            infor[cid][aid][lds_feat_names[2]] += 1
        elif ldcode - min_ldcode == 3:  # past of last two week
            if lds_feat_names[3] not in infor[cid][aid]:
                infor[cid][aid][lds_feat_names[3]] = 0
            infor[cid][aid][lds_feat_names[3]] += 1
    return infor




def convert_df2dict(data, primary_key):
    # sometimes it is more efficient to use dict for extract features
    return data.set_index(primary_key).to_dict(orient='index')


def extract_features_mapping(
        article,
        customer,
        transaction,
        na_filler=-1,
):
    arr_features = extract_arr_features(article, transaction, na_filler)
    arr_features = convert_df2dict(arr_features, 'article_id')
    apk_features = extract_apk_features(customer, transaction, na_filler)
    apk_features = convert_df2dict(apk_features, 'customer_id')
    historical_purchasing_infor = extract_historical_purchased_items(transaction)
    return (
        arr_features,
        apk_features,
        historical_purchasing_infor,
    )


def combine_info_and_get_sampples(
        article_id,
        customer_id,
        article_feature_dict,
        customer_feature_dict,
        interaction_feature_dict,
):
    features = {}
    features.update(article_feature_dict[article_id])
    features.update(customer_feature_dict[customer_id])

    #
    features['buying_same_article_tw'] = 0  # customer buy amount of same article this week
    features['buying_same_article_lw'] = 0
    features['buying_same_article_llw'] = 0
    features['buying_same_article_lllw'] = 0

    # does not purchase any in last three week
    if customer_id not in interaction_feature_dict:
        return features

    # does not buy same article in last three week
    if article_id not in interaction_feature_dict[customer_id]:
        return features

    if 'tw' in interaction_feature_dict[customer_id][article_id]:
        features['buying_same_article_tw'] = interaction_feature_dict[customer_id][article_id]['tw']

    if 'lw' in interaction_feature_dict[customer_id][article_id]:
        features['buying_same_article_lw'] = interaction_feature_dict[customer_id][article_id]['lw']

    if 'llw' in interaction_feature_dict[customer_id][article_id]:
        features['buying_same_article_llw'] = interaction_feature_dict[customer_id][article_id]['llw']

    if 'lllw' in interaction_feature_dict[customer_id][article_id]:
        features['buying_same_article_lllw'] = interaction_feature_dict[customer_id][article_id]['lllw']

    return features