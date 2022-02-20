import numpy as np
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    total = np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
    valid_samples = np.sum([ (a != []) for a in actual])  # 忽略对于没有购买行为的顾客
    return total / valid_samples


def recall4rp(actual, recall):
    if actual is []:
        return -1
    if recall is []:
        return -2
    return len(set(actual) & set(recall)) / len(set(actual))
