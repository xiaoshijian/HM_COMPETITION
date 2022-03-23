import math


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


