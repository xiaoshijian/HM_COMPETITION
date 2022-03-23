


def convert_df2dict(data, primary_key):
    # sometimes it is more efficient to use dict for extract features
    return data.set_index(primary_key).to_dict(orient='index')


def combine_info_and_get_sampples(
        article_id,
        customer_id,
        article_feature_dict,
        customer_feature_dict,
        interaction_feature_dict,
):
    features = {}
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
