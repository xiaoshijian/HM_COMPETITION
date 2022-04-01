import random

def combine_info_for_ranker(
        article_id,
        customer_id,
        article_feature_dict,
        customer_feature_dict,
        interaction_feature_dict,
):
    features = {'customer_id': customer_id}
    features.update(article_feature_dict[article_id])
    features.update(customer_feature_dict[customer_id])

    #
    features['buying_same_article_tw'] = 0  # customer buy amount of same article this week
    features['buying_same_article_lw'] = 0
    features['buying_same_article_llw'] = 0
    features['buying_same_article_lllw'] = 0

    # does not purchase any in last three week
    if customer_id not in interaction_feature_dict:
        return features

    # does not buy same article in last three week
    if article_id not in interaction_feature_dict[customer_id]:
        return features

    if 'tw' in interaction_feature_dict[customer_id][article_id]:
        features['buying_same_article_tw'] = interaction_feature_dict[customer_id][article_id]['tw']

    if 'lw' in interaction_feature_dict[customer_id][article_id]:
        features['buying_same_article_lw'] = interaction_feature_dict[customer_id][article_id]['lw']

    if 'llw' in interaction_feature_dict[customer_id][article_id]:
        features['buying_same_article_llw'] = interaction_feature_dict[customer_id][article_id]['llw']

    if 'lllw' in interaction_feature_dict[customer_id][article_id]:
        features['buying_same_article_lllw'] = interaction_feature_dict[customer_id][article_id]['lllw']

    return features



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

        sample_feature = combine_info_for_ranker(
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



samples4finedrank_trn_cv = generate_samples4finedrank_direct(recall_and_purchased_trn_cv)
samples_feature4finedrank_trn_cv = generate_samples_feature4finedrank(
    samples4finedrank_trn_cv,
    arr_trn_cv,
    apk_trn_cv,
    hpi_trn_cv,
)
# # drop features are not in numeric dataform
samples_feature4finedrank_trn_cv = samples_feature4finedrank_trn_cv.drop(
    ['postal_code'], axis=1
)

# samples4finedrank_trn_lb = generate_samples4finedrank_direct(recall_and_purchased_trn_lb)
