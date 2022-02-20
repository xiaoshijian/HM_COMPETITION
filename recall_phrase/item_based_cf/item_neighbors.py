import sys
sys.path.append('../../')
from toolbox.decorator import timer

@timer
def get_k_most_similar_items(W, k=50):
    # 主要是参照《推荐系统实践》实现的
    k_most_similar_items = dict()
    for i, related_times_info in W.items():
        # 根据相似度，获取k most similar
        sorted_items = sorted(
            related_times_info.items(),
            reverse=True,
            key=lambda x: x[1])  # sort items by similarity
        k_most_similar_items[i] = sorted_items[:k]  # extract k most similar items
    return k_most_similar_items