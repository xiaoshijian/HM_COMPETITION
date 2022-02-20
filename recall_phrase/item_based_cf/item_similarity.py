import math
import sys
sys.path.append('../../')
from toolbox.decorator import timer

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
