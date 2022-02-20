def ibcf_recall(purchased_items, W, K=50, N=50):
    # 这里的pi是对应不同行为的值的，而在这个数据集中，只有购买这个行为，就直接设置为1
    pi = 1
    rank = {}
    for item in purchased_items:
        for neighbor, w_neigh in sorted(
                W[item].items(), key= lambda x: x[1], reverse=True)[0:K]:

            if neighbor in purchased_items:
                continue

            rank[neighbor] += pi * w_neigh


    # sort rank with its similarity
    rank = sorted(rank.items(), key= lambda x: x[1], reverse=True)[0:N]

    # get rank
    rank = [item[0] for item in rank]
    return rank


def run_recall(samples, W, K=50, N=50):
    # user_id_list = train['customer_id'].tolist()
    samples['ibcf_recall'] = samples['transations_list'].apply(
        lambda x: ibcf_recall(x), args=(W, K, N,), axis=1)
    return samples
