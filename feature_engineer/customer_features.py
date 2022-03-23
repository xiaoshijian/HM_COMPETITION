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
