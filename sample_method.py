from dateutil import parser
import datetime

import pandas as pd
import numpy as np


def extract_trn_and_dev(data, perioed, dev_days=7):
    '''
    data: dataframe
    dev_period: list, contains start date and end date
    '''
    # extract sample based on date
    # extract trn_sample from start date to dev_days days before end date
    #         dev_sample from dev_days days before end date to end date
    start_date, end_date = perioed
    end_date_trn = parser.parse(start_date) - datetime.timedelta(days=dev_days)
    end_date_trn = end_date_trn.strftime('%Y-%m-%d')

    trn_condition = '(t_dat >= "{}" and t_dat < "{}")'.format(start_date, end_date_trn)
    dev_condition = '~(t_dat >= "{}" and t_dat < "{}")'.format(end_date_trn, end_date)
    trn_data = data.query(trn_condition).reset_index(drop=True)
    dev_data = data.query(dev_condition).reset_index(drop=True)
    return trn_data, dev_data

def get_dcode_mapping(dates):
    # 一个日期对应数字的mapping，能加快计算

    '''
    convert dates into dcode, which is efficient for computing
    :param dates: transaction days in dateform
    :return:
    '''
    date_nums = len(dates.unique())
    d2c_mapping = {}
    c2d_mapping = {}
    max_date = dates.max()
    for i in range(date_nums):
        cur_date = max_date - pd.Timedelta(days=i)
        d2c_mapping[cur_date.strftime('%Y-%m-%d')] = i
        c2d_mapping[i] = cur_date.strftime('%Y-%m-%d')
    return d2c_mapping, c2d_mapping


def get_dcode_and_ldcode(dataframe, d2c_mapping):
    '''
    convert date into dcode and get ldcode with dcode, default to set 7 days into one session with dcode
    :param dataframe: transactions
    :param d2c_mapping: date2code mapping
    :return: transactions
    '''

    dataframe['dcode'] = dataframe['t_dat'].apply(lambda x: d2c_mapping[x] if x in d2c_mapping else np.nan)
    dataframe['ldcode'] = dataframe['dcode'] // 7
    dataframe = dataframe.drop(columns=['t_dat', 't_dat_dateform'])
    return dataframe


def extract_tran4feat(
        dataframe,
        ld4feat,
        d2c_mapping,
        tran4feat_sessions=28,
        is_tran4model=False,
):
    '''
    :param dataframe: transactions
    :param ld4feat: last date for transaction to generating features
    :param d2c_mapping: date2code mapping
    :param tran4feat_sessions: how long session for transaction to generate features
    :param is_tran4model: if output transaction for building model, if True, the session lasting for
                          7 days after transactions
    :return:
    '''

    ld4feat_code = d2c_mapping[ld4feat]
    tran4feat_cond = (
        ((dataframe['dcode'] - ld4feat_code) >= 0) & \
        ((dataframe['dcode'] - ld4feat_code) < tran4feat_sessions)
    )
    tran4feat = dataframe[tran4feat_cond].reset_index(drop=True)

    tran4model = None
    if is_tran4model == True:
        tran4model_cond = (
            (ld4feat_code - dataframe['dcode'] >= 0) & \
            (ld4feat_code - dataframe['dcode'] < 7)

        )
        tran4model = dataframe[tran4model_cond].reset_index(drop=True)
    return (tran4feat, tran4model)






