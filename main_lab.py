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

# 这玩意召回的质量堪忧啊
def generate_fined_rank_training_samples(samples, p2n_ratio=3):
    for purchased_items, rwp_items in zip(samples['purchased_items'],
                                         samples['recall_without_purchased_items']):
        pass
    return 



# sample_trn['total_recall'] = 
