
'''
mainly useing for get interacting features
'''
def extract_historical_purchased_items(transaction):
    min_ldcode = transaction['ldcode'].min()
    lds_feat_names = ['tw', 'lw', 'llw', 'lllw']

    infor = {}
    for cid, aid, ldcode in zip(
        transaction['customer_id'],
        transaction['article_id'],
        transaction['ldcode']
    ):
        if cid not in infor:
            infor[cid] = {}
        if aid not in infor[cid]:
            infor[cid][aid] = {}
        if ldcode - min_ldcode == 0:
            if lds_feat_names[0] not in infor[cid][aid]: # this week
                infor[cid][aid][lds_feat_names[0]] = 0
            infor[cid][aid][lds_feat_names[0]] += 1
        elif ldcode - min_ldcode == 1:  # last week
            if lds_feat_names[1] not in infor[cid][aid]:
                infor[cid][aid][lds_feat_names[1]] = 0
            infor[cid][aid][lds_feat_names[1]] += 1
        elif ldcode - min_ldcode == 2:  # past of last week
            if lds_feat_names[2] not in infor[cid][aid]:
                infor[cid][aid][lds_feat_names[2]] = 0
            infor[cid][aid][lds_feat_names[2]] += 1
        elif ldcode - min_ldcode == 3:  # past of last two week
            if lds_feat_names[3] not in infor[cid][aid]:
                infor[cid][aid][lds_feat_names[3]] = 0
            infor[cid][aid][lds_feat_names[3]] += 1
    return infor

