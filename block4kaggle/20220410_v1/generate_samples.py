def extract_tran4model(
        dataframe,
        ldb4model,
        d2c_mapping,
        tran4feat_sessions=28,
        tran4recall_sessions=21,
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
    ldb4model_code = d2c_mapping[ldb4model]
    tran4feat_cond = (
            ((dataframe['dcode'] - ldb4model_code) >= 0) \
            & ((dataframe['dcode'] - ldb4model_code) < tran4feat_sessions)
    )
    tran4feat = dataframe[tran4feat_cond].reset_index(drop=True)

    tran4recall_cond = (
            ((dataframe['dcode'] - ldb4model_code) >= 0) \
            & ((dataframe['dcode'] - ldb4model_code) < tran4recall_sessions)
    )
    tran4recall = dataframe[tran4recall_cond].reset_index(drop=True)

    tran4model = None
    if is_tran4model == True:
        tran4model_cond = (
                (ldb4model_code - dataframe['dcode'] > 0) \
                & (ldb4model_code - dataframe['dcode'] <= 7)  # it is wrong here
        )
        tran4model = dataframe[tran4model_cond].reset_index(drop=True)
    return (tran4feat, tran4recall, tran4model)


def seperate_hot_and_cold(tran4feat, customer, tran4model=None):
    hot_customer = set(tran4feat.customer_id.unique())
    cold_customer = set(customer.customer_id.tolist()) - hot_customer

    hot4model = None
    cold4model = None
    if tran4model:
        hot4model = tran4model.query('customer_id.isin(@hot_customer)').reset_index()
        cold4model = tran4model.query('customer_id.isin(@cold_customer)').reset_index()
    return hot_customer, cold_customer, hot4model, cold4model


if __name__ == '__main__':
    hot_customer_trn_cv, cold_customer_trn_cv, hot4model_trn_cv, cold4model_trn_cv = seperate_hot_and_cold(
        tran4recall_trn_cv,
        customer,
        tran4model_trn_cv,
    )

    hot_customer_dev_cv, cold_customer_dev_cv, hot4model_dev_cv, cold4model_dev_cv = seperate_hot_and_cold(
        tran4recall_dev_cv,
        customer,
        tran4model_dev_cv,
    )
