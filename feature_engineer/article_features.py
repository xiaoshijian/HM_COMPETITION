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

