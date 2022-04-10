!pip install bayesian-optimization
from bayes_opt import BayesianOptimization
BayesianOptimization
def run_bayes_opt(recall_and_purchased,
              p2n_ratio,
              arr_trn,
              apk_trn,
              hpi_trn,
              recall_and_purchased_dev,
              arr_dev,
              apk_dev,
              hpi_dev,
              init_points=5,
              n_iter=10,
    ):
    samples4finedrank = generate_samples4finedrank_direct(
        recall_and_purchased,
        p2n_ratio
    )
    samples_feature4finedrank = generate_samples_feature4finedrank(
        samples4finedrank,
        arr_trn,
        apk_trn,
        hpi_trn,
    )
    pbounds = {
        'num_leaves': (10, 100),
        'max_depth': (3, 20),
        'learning_rate': (0.01, 1),
        'n_estimators': (500, 2000),
        'subsample_for_bin': (100000, 500000),
        'min_child_weight': (0.001, 0.1),
        'subsample': (0.7, 1),
        'subsample_freq': (0, 1),
        'colsample_bytree': (0.7, 1),
        'reg_alpha': (0, 5),
        'reg_lambda': (0, 5),
        }
    def black_box_function(n_estimators,
                           max_depth,
                           num_leaves,
                           learning_rate,
                           subsample_for_bin,
                           min_child_weight,
                           subsample_freq,
                           subsample,
                           colsample_bytree,
                           reg_alpha,
                           reg_lambda,
                           ):
        bayopt_paras = {
            'num_leaves': int(num_leaves),
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'n_estimators': int(n_estimators),
            'subsample_for_bin': int(subsample_for_bin),
            'min_child_weight': min_child_weight,
            'subsample': min(1, (subsample * 100 + 3) // 5 * 0.05 ),
            'subsample_freq': subsample_freq,
            'colsample_bytree': min(1, (colsample_bytree * 100 + 3) // 5 * 0.05),
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            }
        selected_features = feature_selections_with_mv_and_tree(
            samples=samples_feature4finedrank,
            paras=bayopt_paras,
        )
        clf = train_tree_model_for_ranker(
            data=samples_feature4finedrank,
            selected_features=selected_features,
            customer_id='customer_id',
            label='label',
            mode='lgb',
            paras=bayopt_paras,
        )

        prediction_dev = pred(
            recall_and_purchased_dev,
            clf,
            arr_dev,
            apk_dev,
            hpi_dev,
            selected_features,
        )
        eval_info = helper(recall_and_purchased_dev, prediction_dev)
        score = mapk(eval_info['purchased_items'], eval_info['recommend'])
        return score

    optimizer = BayesianOptimization(f=black_box_function,
                                     pbounds=pbounds,
                                     random_state=1992,
                                     )
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )
    return optimizer

"""
below is code
"""
p2n_ratio=4
run_bayes_opt(
    hot_recall_and_purchased_trn_cv,
    p2n_ratio,
    arr_trn_cv,
    apk_trn_cv,
    hpi_trn_cv,
    hot_recall_and_purchased_dev_cv,
    arr_dev_cv,
    apk_dev_cv,
    hpi_dev_cv,
    init_points=5,
    n_iter=10,
)

