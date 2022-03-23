
import pandas as pd
import numpy as np

#
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


def extract_tran4ibcf(
        dataframe,
        ld4ibcf,
        d2c_mapping,
        tran4ibcf_sessions=21,
):
    '''
    :param dataframe: transactions
    :param ld4feat: last date for transaction to get item base colloborative filtering
    :param d2c_mapping: date2code mapping
    :param tran4ibcf_sessions: how long session for transaction to generate features
    :return:
    '''
    ld4ibcf_code = d2c_mapping[ld4ibcf]
    tran4ibcf_cond = (
        ((dataframe['dcode'] - ld4ibcf_code) >= 0) & \
        ((dataframe['dcode'] - ld4ibcf_code) < tran4ibcf_sessions)
    )
    tran4ibcf = dataframe[tran4ibcf_cond].reset_index(drop=True)
    return tran4ibcf


