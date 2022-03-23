from .item_based_cf.recall import get_recall_from_transaction_by_ibcf
from .majority.recall import get_majority_in_l3w

def get_recall_by_ibcf_and_majority(transactions):

    # W and recall by ibcf
    W4ibcf, cpi4recall = get_recall_from_transaction_by_ibcf(transactions)
    _, __, ___, majority = get_majority_in_l3w(transactions)  # get recall by majority

    cpi4recall['recall_by_majority'] = [majority for _ in range(len(cpi4recall))]
    cpi4recall['recall_items'] = cpi4recall.apply(
        lambda x: list(set([item[0] for item in x.recall_by_ibcf]) | set(x.recall_by_majority), axis=1 )
    )  # merge them
    return (W4ibcf, cpi4recall)

