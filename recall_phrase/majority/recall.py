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
