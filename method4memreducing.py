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

# d2c_mapping, d2c_mapping = get_dcode_mapping(dates)
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
