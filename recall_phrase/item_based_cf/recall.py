

from copy import copy, deepcopy
from .calculate_similarity import cal_item_similariy_with_time_weighting

def add_adl(transactions):
    # concat article_id, dcode and ldcode
    transactions['adl'] = transactions.apply(
        lambda x: (x.article_id, x.dcode, x.ldcode), axis=1
    )
    return transactions


def get_customer_purchasing_items(transactions):
    transactions_cpy = deepcopy(transactions)
    transactions_cpy = add_adl(transactions_cpy)
    transactions_cpy = transactions_cpy.groupby('customer_id')['adl'].apply(list).reset_index(name='purchased_items')
    return transactions_cpy


def weight_item_by_ldcode(ldcode, min_ldcode):
    # session_split的命名可以优化
    diff = ldcode - min_ldcode
    return 1 / (1 + diff) if diff <= 2 else 0


def ibcf_recall_method2(purchased_items, W, min_ldcode, N=50):
    # 使用三个星期的购买记录作为召回的items，其中tw的weight是1, lw的weight是1/2， llw的weight是1/3
    # 使用多种不同的weight进行召回
    # 可能在2个小时左右
    # set weight
    recall_items = {}
    for item, _, ldcode in purchased_items:
        if item not in W:
            continue
        date_weight = weight_item_by_ldcode(ldcode, min_ldcode)
        # short cut for those items are too remote
        if date_weight == 0:
            continue
        for nbor, nbor_weight in W[item].items():
            if nbor not in recall_items:
                recall_items[nbor] = 0
            recall_items[nbor] += date_weight * nbor_weight
    recall_items = sorted(recall_items.items(), key = lambda x: x[1], reverse=True)
    # recall_items = [item[0] for item in recall_items]
    return recall_items[:N]


def get_recall_from_transaction_by_ibcf(transactions):
    W4ibcf = cal_item_similariy_with_time_weighting(transactions)
    min_ldcode = transactions.ldcode.min()

    # cpi is abbrevation for customer purchasing items
    cpi4recall = get_customer_purchasing_items(transactions)
    cpi4recall['recall_by_ibcf'] = cpi4recall['purchased_items'].apply(
        ibcf_recall_method2, args=(W4ibcf, min_ldcode,)
    )
    cpi4recall = cpi4recall[['customer_id', 'recall_by_ibcf']]
    return (W4ibcf, cpi4recall)
