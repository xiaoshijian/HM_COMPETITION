def feature_selections_with_mv_and_tree(samples, p4mv=0.05, ratio4tree=0.6, paras=None):
    mv4features = get_mv_for_features(
        data=samples,
        customer_id='customer_id',
        label='label',
        paras=paras,
    )

    selected_features_from_mv = []
    for feature, info in mv4features.items():
        if info['p-value'][1] <= p4mv:
            selected_features_from_mv.append(feature)

    print('feature from mv test for customer: ')
    print(selected_features_from_mv)

    _, feature_importance_from_tree, features_and_importance_from_tree = get_importance_from_ranker(
        data=samples,
        customer_id='customer_id',
        label='label',
        mode='lgb',
    )
    features_name = [col for col in samples.columns if col not in ['label', 'customer_id']]

    features_and_importance_from_tree = sorted(features_and_importance_from_tree, key=lambda x: x[1], reverse=True)
    selected_features_from_tree = [
        item[0] for item in features_and_importance_from_tree[:int(ratio4tree * len(features_and_importance_from_tree))]
    ]
    print('feature from lgb: ')
    print(selected_features_from_tree)

    selected_features = list(set(selected_features_from_mv) | set(selected_features_from_tree))

    for item in ['FN', 'Active']:
        if item not in selected_features:
            selected_features.append(item)
    return selected_features


