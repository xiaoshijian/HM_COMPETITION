import pandas as pd
import math
from copy import copy, deepcopy

def cal_weight_with_time(days_diff, K=1, T=15):
    # weight with decaying time diff
    # formular based on https://stats.stackexchange.com/questions/196653/assigning-more-weight-to-more-recent-observations-in-regression
    weight = K * math.exp(-days_diff / T)
    return weight


def cal_ij_simi_time_weighting(cooc_info, K=1, T=15):
    simi = sum([cal_weight_with_time(abs(x[1] - x[0]), K, T)
                for x in cooc_info])  # plus one avoid divide by zero
    return simi


def cal_item_similariy_with_time_weighting(transaction, mode='time_weighting', K=200):
    # min_dcode = transaction['dcode'].min()
    info4similarity = transaction[['customer_id', 'article_id', 'dcode']]
    info4similarity['aid_n_dcode'] = info4similarity.apply(
        lambda x: (x.article_id, x.dcode), axis=1)
    info4similarity = info4similarity.groupby('customer_id')['aid_n_dcode'].apply(list).reset_index(
        name='historical_behaviors')
    info4similarity['historical_behaviors'] = info4similarity['historical_behaviors'].apply(
        lambda x: sorted(x, key=lambda x: x[1], reverse=False)
    )

    C = dict()
    N = dict()
    for cid, hb in zip(
            info4similarity['customer_id'],
            info4similarity['historical_behaviors']
    ):
        for aid_i, dcode_i in hb:
            if aid_i not in N:
                N[aid_i] = 0
            N[aid_i] += 1
            if aid_i not in C:
                C[aid_i] = {}
            for aid_j, dcode_j in hb:
                if aid_j == aid_i:
                    continue
                if aid_j not in C[aid_i]:
                    C[aid_i][aid_j] = []
                C[aid_i][aid_j].append((dcode_i, dcode_j))  # 记录aid_i, aid_j的时间

    W = {}
    for i, related_items in C.items():
        if i not in W:
            W[i] = {}
        for j, cooc_info in related_items.items():
            if mode == 'time_weighting':
                W[i][j] = cal_ij_simi_time_weighting(cooc_info)

    # for each item, only save items with largest k similarity
    W_with_largest_k_similarity = {}
    for i, i_nbors in W.items():
        W_with_largest_k_similarity[i] = {}
        largest_k_similarity = sorted(i_nbors.items(), key=lambda x: x[1], reverse=True)[:K]
        for j, j_simi in largest_k_similarity:
            W_with_largest_k_similarity[i][j] = j_simi
    return W_with_largest_k_similarity
    # return W


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




def get_majority_in_l3w(transactions, n=30):
    min_ldcode = transactions.ldcode.min()
    tw_condition = (transactions.ldcode - min_ldcode == 0)
    lw_condition = (transactions.ldcode - min_ldcode == 1)
    llw_condition = (transactions.ldcode - min_ldcode == 2)
    majority_tw = transactions[tw_condition].article_id.value_counts().index[:n]
    majority_lw = transactions[lw_condition].article_id.value_counts().index[:n]
    majority_llw = transactions[llw_condition].article_id.value_counts().index[:n]

    majority = list(set(majority_tw) | set(majority_lw) | set(majority_llw))

    return majority_tw, majority_lw, majority_llw, majority


def repurchase_recall(transactions, timespan_ldcode=2):
    min_ldcode = transactions['ldcode'].min()
    subsample = transactions[transactions['ldcode'] - min_ldcode <= timespan_ldcode]
    repurchase_recall = subsample.groupby('customer_id')['article_id'].apply(list).reset_index(name='recall_by_repurchased')
    return repurchase_recall



def get_recall_by_multi_methods(transactions):
    # W and recall by ibcf
    W4ibcf, cpi4recall = get_recall_from_transaction_by_ibcf(transactions)
    _, __, ___, majority = get_majority_in_l3w(transactions)  # get recall by majority
    repurchase = repurchase_recall(transactions)

    cpi4recall['recall_by_majority'] = [majority for _ in range(len(cpi4recall))]
    cpi4recall = pd.merge(
        cpi4recall,
        repurchase,
        how='left',
        on='customer_id'
    )

    cpi4recall['recall_items'] = cpi4recall.apply(
        lambda x: list(set([item[0] for item in x.recall_by_ibcf]) \
                       | set(x.recall_by_majority) \
                       | set(x.recall_by_repurchased)),
        axis=1
    )  # merge them
    return (W4ibcf, cpi4recall)



def combine_recall_and_purchasing_items_nw(
        recall_items,
        purchased_items,
):
    majority = recall_items['recall_by_majority'][0][: ],  # setting for those not recall
    # purchased_items = get_customer_purchasing_items(purchased_items)
    purchased_items['purchased_items'] = purchased_items['purchased_items'].apply(
        lambda x: [item[0] for item in x]
    )
    output = pd.merge(purchased_items,
                      recall_items[['customer_id', 'recall_items']],
                      how='left',
                      on='customer_id')

    # fill na in recall_items with majority, random recommend by majority algorithm
    output['recall_items'] = output['recall_items'].apply(lambda d: d if isinstance(d, list) else majority[:])
    # recall but not purchase
    output['recall_without_purchased_items'] = output.apply(
        lambda x: list(set(x.recall_items) - set(x.purchased_items)), axis=1
    )
    return output