import pandas as pd
import random
from .feature_engineer.method4fe import combine_info_and_get_sampples

def generate_samples4finedrank_direct(samples, p2n_ratio=3):
    # 直接生成带label的
    samples4finedrank = []
    for customer_id, purchased_items, rwp_items in zip(samples['customer_id'],
                                                       samples['purchased_items'],
                                                       samples['recall_without_purchased_items']):
        for item in purchased_items:
            # random select sample from recall items and build up a samples pairs

            # 导入正样本
            samples4finedrank.append({'customer_id': customer_id,
                                      'article_id': item,
                                      'label': 1})
            selected_rwp_items = random.sample(rwp_items, p2n_ratio)
            for rwp_item in selected_rwp_items:
                # 导入负样本
                samples4finedrank.append({'customer_id': customer_id,
                                          'article_id': rwp_item,
                                          'label': 0})
    return samples4finedrank


def generate_samples_feature4finedrank(samples4finedrank,
                                       article_feature_dict,
                                       customer_feature_dict,
                                       interaction_feature_dict):
    output = []
    for sample in samples4finedrank:
        label, customer_id, article_id = sample['label'], sample['customer_id'], sample['article_id']

        sample_feature = combine_info_and_get_sampples(
            article_id,
            customer_id,
            article_feature_dict,
            customer_feature_dict,
            interaction_feature_dict
        )
        sample_feature['label'] = label
        output.append(sample_feature)
    output = pd.DataFrame(output)
    return output


def generate_samples4infer(customers4infer,
                           recall_items):
    recall_by_majority = [item for item in recall_items['recall_by_majority'][0]]  # get majority
    output = pd.merge(
        customers4infer,
        recall_items,
        how='left',
        on='customer_id'
    )

    output = output[['customer_id', 'recall_items']]
    output['recall_items'] = output['recall_items'].apply(
        lambda x: x if isinstance(x, list) else recall_by_majority
    )
    return output






