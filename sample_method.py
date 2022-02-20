from dateutil import parser
import datetime


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


