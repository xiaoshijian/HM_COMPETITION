import math
import sys
sys.path.append('../../')
from toolbox.decorator import timer



def cal_ij_simi_time_weighting(cooc_info, k=-1):
    simi = sum([(abs(x[1] - x[0]) + 1) ** k for x in cooc_info])  # plus one avoid divide by zero
    return simi

@timer
def cal_item_similariy(transations_list, mode='standard'):
    # 主要是参照《推荐系统实践》实现的
    assert mode in ['standard', 'avoid_pop']
    C = dict()
    N = dict()
    for items in transations_list:
        for i in items:
            if i not in N:
                N[i] = 0
            N[i] += 1
            if i not in C:
                C[i] = {}
            for j in items:
                if j == i:
                    continue
                if j not in C[i]:
                    C[i][j] = 0
                C[i][j] += 1
    W = {}
    for i, related_items in C.items():
        if i not in W:
            W[i] = {}
        for j, c_ij in related_items.items():
            if mode == 'standard':
                W[i][j] = c_ij / N[i]
            elif mode == 'avoid_pop':
                W[i][j] = c_ij / math.sqrt(N[i] * N[j])
    return W




def cal_item_similariy_with_time_weighting(transaction, mode='time_weighting'):
    # min_dcode = transaction['dcode'].min()
    info4similarity = transaction[['customer_id', 'article_id', 'dcode']]
    info4similarity['aid_n_dcode'] = info4similarity.apply(
        lambda x: (x.article_id, x.decode), axis=1)
    info4similarity = info4similarity.groupby('customer_id')['aid_n_dcode'].apply(list).reset_index(name='historical_behaviors')
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
    return W

