from copy import copy 
import pandas as pd
sample_trn = deepcopy(purchased_item_session_list[1][2])
sample_trn = pd.merge(sample_trn,
                      samples4recall[['customer_id', 'ibcf_recall']],
                      on='customer_id',
                      how='left')


item_count = {}
for items in purchased_item_session_list[-1][2].purchased_items:
    for item in items:
        if item not in item_count:
            item_count[item] = 0
        item_count[item] += 1
        
item_count = sorted(item_count.items(), key = lambda x: x[1], reverse = False)
most_frequency_items_last_week = [item[0] for item in item_count][:30]



sample_trn['majority_recall'] = [most_frequency_items_last_week[:] for _ in range(len(sample_trn))]


def merge_multi_recall_method(recall_method_list):
    total_recall = []
    for recall_method in recall_method_list:
        if not isinstance(recall_method, list):   # 这里不是list的话，就为空na，就是没有召回
            continue 
        for item in recall_method:
            if item not in total_recall:
                total_recall.append(item)
    return total_recall

sample_trn['total_recall'] = sample_trn.apply(
    lambda x: merge_multi_recall_method(
        [x.ibcf_recall, x.majority_recall]
    ), axis = 1
)
sample_trn['recall_without_purchased_items'] = sample_trn.apply(
    lambda x: list(set(x.total_recall) - set(x.purchased_items)) , axis = 1
)

def generate_fined_rank_training_samples(samples, p2n_ratio=2):
    fined_rank_samples = []
    for customer_id, purchased_items, rwp_items in zip(samples['customer_id'],
                                                      samples['purchased_items'],
                                                      samples['recall_without_purchased_items']):
        for item in purchased_items:
            # random select sample from recall items and build up a samples pairs
            selected_rwp_items = random.sample(rwp_items, p2n_ratio)
            for rwp_item in selected_rwp_items:
                fined_rank_samples.append({'customer_id': customer_id,
                                           'positice_article': item,
                                           'negative_article': rwp_item})
        
    return fined_rank_samples

samples4finedrank = generate_fined_rank_training_samples(sample_trn)



# drop_duplicat
from dateutil import parser
import datetime


def split_transaction_into_session(data, end_date, session_last=7, session_nums=10):
    '''
    iterate for end_date to get session nums
    '''
    date2end = end_date
    session_list = []
    for _ in range(session_nums):
        date2start = parser.parse(date2end) - datetime.timedelta(days=session_last)
        date2start = date2start.strftime('%Y-%m-%d')
        session_condition = '(t_dat >= "{}" and t_dat < "{}")'.format(date2start, date2end)
        session = data.query(session_condition).reset_index(drop=True)
        session_list.append([date2start, date2end, session])
        date2end = date2start
    return session_list


def get_purchased_item_session(data):
    customer_behavior = data.groupby('customer_id')['article_id'].apply(list).reset_index(name='purchased_items')
    return customer_behavior


session_list = split_transaction_into_session(
    data=transactions,
    end_date='2020-09-23',
    session_nums=3
) 

purchased_item_session_list = [[item[0], item[1], get_purchased_item_session(item[2])] for item in session_list]



# 如果可以使用recall来recall的，就使用recall，如果不能的话，就用majority





# I think original running performace for mv test is too slow
# the main point should be on the comparison between pr and pr
# I try to youhua it by using leetcode in question2



'''
it is not yet implement and check
'''


import numpy as np
def last_position_of_each_elements(xy):
    # 计算以x中每个值的最后的位置
    xy = xy[xy[:, 0].argsort()]
    x = xy[:, 0]
    size = len(x)
    value_set = set()
    output = []
    for i in range(len(x)-1, 0):
        value = x[i]
        if value not in value_set:
            value_set.add(value)
            output.append((value, (i + 1) / size))
    return output[::-1]


def cal_partial_result(pr, x_in_xy, x_in_xyr):
    # x_in_xy, x_in_xyr均为list，其中的元素是（value，position_in_percentage）
    l1 = len(x_in_xy)
    l2 = len(x_in_xyr)
    p1, p2 = 0, 0

    partial_result = 0
    while p2 < l2 and p1 < l1:
        # 对于x_in_xy中的每一个元素，寻找在它在x_in_xyr的位置，并找出它的前一个元素

        if x_in_xy[p1][0] > x_in_xyr[p2][0]:
            while p2 < l2 and x_in_xy[p1][0] > x_in_xyr[p2][0]:
                p2 += 1
            ratio1 = x_in_xy[p1][1]
            ratio2 = x_in_xyr[p2-1][1]
            partial_result += pr * np.square(ratio1 - ratio2)
        elif x_in_xy[p1][0] > x_in_xyr[p2][0]:
            ratio1 = x_in_xy[p1][1]
            ratio2 = x_in_xyr[p2][1]
            partial_result += pr * np.square(ratio1 - ratio2)
            p2 += 1
        else:
            if p2 == 0:
                ratio1 = x_in_xy[p1][1]
                partial_result += pr * np.square(ratio1 - 0)
            else:
                ratio1 = x_in_xy[p1][1]
                ratio2 = x_in_xyr[p2 - 1][1]
                partial_result += pr * np.square(ratio1 - ratio2)
        p1 += 1

    for p in range(p1, l1):
        ratio1 = x_in_xy[p1][1]
        partial_result += pr * np.square(ratio1 - 1)
    return partial_result

