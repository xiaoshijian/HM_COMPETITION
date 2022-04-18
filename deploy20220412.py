import math

#
def cal_ibcf_values_for_customers(transactions, N=20):
    W4ibcf, cpi4recall = get_recall_from_transaction_by_ibcf(transactions)
    min_ldcode = transactions.ldcode.min()
    cpi4recall = get_customer_purchasing_items(transactions)
    cpi4recall['recall_by_ibcf'] = cpi4recall['purchased_items'].apply(
        ibcf_recall_method2, args=(W4ibcf, min_ldcode, N,)
    )
    cpi4recall = cpi4recall[['customer_id', 'recall_by_ibcf']]
    ibcf_value_dict = {}
    for cid, items in zip(cpi4recall['customer_id'], cpi4recall['recall_by_ibcf']):
        ibcf_value_dict[cid] = {}
        for item in items:
            ibcf_value_dict[item[0]] = item[1]
    return ibcf_value_dict


def generate_feature_for_ranker_v2(
        article_id,
        customer_id,
        article_feature_dict,
        customer_feature_dict,
        interaction_feature_dict,
        ibcf_values_dict,
):
    '''
    use
    article_feature_dict, customer_feature_dict,
    interaction_feature_dict, ibcf_values_dict

    '''
    features = {'customer_id': customer_id}
    features.update(article_feature_dict[article_id])
    features.update(customer_feature_dict[customer_id])

    #
    features['buying_same_article_tw'] = 0  # customer buy amount of same article this week
    features['buying_same_article_lw'] = 0
    features['buying_same_article_llw'] = 0
    features['buying_same_article_lllw'] = 0

    # does not purchase any in last three week
    if customer_id not in interaction_feature_dict:
        pass
    # does not buy same article in last three week
    elif article_id not in interaction_feature_dict[customer_id]:
        pass
    else:
        if 'tw' in interaction_feature_dict[customer_id][article_id]:
            features['buying_same_article_tw'] = interaction_feature_dict[customer_id][article_id]['tw']

        if 'lw' in interaction_feature_dict[customer_id][article_id]:
            features['buying_same_article_lw'] = interaction_feature_dict[customer_id][article_id]['lw']

        if 'llw' in interaction_feature_dict[customer_id][article_id]:
            features['buying_same_article_llw'] = interaction_feature_dict[customer_id][article_id]['llw']

        if 'lllw' in interaction_feature_dict[customer_id][article_id]:
            features['buying_same_article_lllw'] = interaction_feature_dict[customer_id][article_id]['lllw']

    # if customer_id not in ibcf_values_dict
    if customer_id not in ibcf_values_dict:
        features['ibcf_values'] = -1
    # if article_id not in ibcf_values_dict[customer_id]
    elif customer_id in ibcf_values_dict and article_id not in ibcf_values_dict[customer_id]:
        features['ibcf_values'] = -1
    else:
        features['ibcf_values'] = ibcf_values_dict[customer_id][article_id]
    return features



# 存在同时购买多件物品，并且退回的情况
# need to check buying same items in one days
def count_buying_same_items_in_one_days(transactions):
    transactions = transactions.groupby(
        by=['customer_id', 'article_id', 'dcode'],
    ).size()
    return transactions


def drop_same_items_in_one_days(transactions):
    transactions = transactions.drop_duplicates(
        subset=['customer_id', 'article_id', 'dcode'], keep='first'
    ).reset_index(drop=True)
    return transactions

def cal_popular_method(x):
    return 1.0 / (math.log(x) + 1)

def get_customer_activities_degree(transactions):
    transactions = transactions.groupby(['customer_id']).size().reset_index(name='customer_active_degree')
    transactions['customer_active_degree'] = transactions['customer_active_degree'].apply(cal_popular_method)
    cad_dict = {}
    for cid, degree in zip(transactions['customer_id'], transactions['customer_active_degree']):
        cad_dict[cid] = degree
    return cad_dict


def get_info4similarity(transactions):
    info4similarity = transactions[['customer_id', 'article_id', 'dcode']]
    info4similarity['aid_n_dcode'] = info4similarity.apply(
        lambda x: (x.article_id, x.dcode), axis=1)
    info4similarity = info4similarity.groupby('customer_id')['aid_n_dcode'].apply(list).reset_index(
        name='historical_behaviors')
    info4similarity['historical_behaviors'] = info4similarity['historical_behaviors'].apply(
        lambda x: sorted(x, key=lambda x: x[1], reverse=False)
    )
    return info4similarity


def cal_similarity_with_customer_activities_degree(transactions):
    transactions = drop_same_items_in_one_days(transactions)
    cad_dict = get_customer_activities_degree(transactions)

    C = dict()
    N = dict()  # sum of w_{u}
    for cid, aid in zip(transactions['customer_id'], transactions['article_id']):
        if cid not in N:
            N[aid] = 0
        N[aid] += cad_dict[cid]

    info4similarity = get_info4similarity(transactions)

    for cid, hb in zip(
            info4similarity['customer_id'],
            info4similarity['historical_behaviors']
    ):
        for aid_i, dcode_i in hb:
            if aid_i not in N:
                N[aid_i] = 0
            N[aid_i] += cad_dict[cid]
            if aid_i not in C:
                C[aid_i] = {}
            for aid_j, dcode_j in hb:
                if aid_j == aid_i:
                    continue
                if aid_j not in C[aid_i]:
                    C[aid_i][aid_j] = []
                C[aid_i][aid_j].append((dcode_i, dcode_j))  # 记录aid_i, aid_j的时间
    return



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

