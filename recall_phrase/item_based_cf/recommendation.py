



def weight_item_by_dcode(dcode, min_dcode):
    # reduce weight while date is remote
    return 1 / ((dcode - min_dcode) ** 0.99)

def weight_item_by_ldcode(ldcode, min_ldcode):
    # session_split的命名可以优化
    # calculate items weight in recall phrase, set weight to 1 while items purchased in this week
    # set 1/2 while in last week and 1/3 while in last last week
    diff = ldcode - min_ldcode
    return 1 / (1 + diff) if diff <= 2 else 0

def ibcf_recall_method1(purchased_items, W, min_decode, N=50):
    # 使用三个星期的购买记录作为召回的items，其中tw的weight是1, lw的weight是1/2， llw的weight是1/3
    # 使用多种不同的weight进行召回
    # 可能在2个小时左右
    # set weight    
    recall_items = {}
    for item, dcode in purchased_items:
        if item not in W:
            continue
        date_weight = weight_item_by_dcode(dcode, min_decode)
        for nbor, nbor_weight in W[item]:
            if nbor not in recall_items:
                recall_items[nbor] += date_weight * nbor_weight
    recall_items = sorted(recall_items.items(), key = lambda x: x[1], reverse=True)
    return recall_items[:N]

def ibcf_recall_method2(purchased_items, W, min_ldcode, N=50):
    # 使用三个星期的购买记录作为召回的items，其中tw的weight是1, lw的weight是1/2， llw的weight是1/3
    # 使用多种不同的weight进行召回
    # 可能在2个小时左右
    # set weight    
    recall_items = {}
    for item, ldcode in purchased_items:
        if item not in W:
            continue
        date_weight = weight_item_by_ldcode(ldcode, min_ldcode)
        for nbor, nbor_weight in W[item]:
            if nbor not in recall_items:
                recall_items[nbor] += date_weight * nbor_weight
    recall_items = sorted(recall_items.items(), key = lambda x: x[1], reverse=True)
    return recall_items[:N]


