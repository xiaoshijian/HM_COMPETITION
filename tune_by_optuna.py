from sklearn.metrics import roc_auc_score
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import StratifiedKFold
import optuna
import lightgbm as lgbm
import xgboost as xgb

def select_best_threshold(actual_label, pred_prob, thresholds_list):
    best_threshold, best_score = None, -1
    for threshold in thresholds_list:
        pred_label = np.where(pred_prob >= threshold, 1, 0)
        auc_on_dev = roc_auc_score(actual_label, pred_label)
        if auc_on_dev > best_score:
            best_score = auc_on_dev
            best_threshold = threshold
    print("best score: {}, corresponding threshold: {}".format(best_score, best_threshold))
    return best_threshold, best_score


def objective4lgb(trial, X, y):
    # 参数网格
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 1000),
        "alpha": trial.suggest_float("alpha", 0, 10),
        "lambda": trial.suggest_float("lambda", 0, 10),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1, step=0.1),
#         "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1, step=0.1),
        "colsample_bytree": trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]), 
        "colsample_bynode": trial.suggest_categorical('colsample_bynode', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        "random_state": 2021,
        
    }
    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1992)
    cv_scores = np.empty(5)
    print('---------------------------begin to run performance on cross validation---------------------------')
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # LGBM建模
        model = lgbm.LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
#             eval_set=[(X_test, y_test)],   # print太多的东西了，麻烦
#             eval_metric="binary_logloss",   # print太多的东西了，麻烦
#             early_stopping_rounds=100,
#             callbacks=[
#                 LightGBMPruningCallback(trial, "binary_logloss")
#             ],
        )
        # 模型预测
        preds = model.predict_proba(X_test)[:,1]
        # 优化指标logloss最小, 这里需要不断地去修改, 需要目标函数去改
        thresholds_list = [i * 0.005 for i in range(200)]
        best_threshold, best_score = select_best_threshold(y_test, preds, thresholds_list)
        cv_scores[idx] = best_score
        print('round{}: {}'.format(idx, best_score))
    print('average performance on cross validation: {}'.format(np.mean(cv_scores)))
    print('---------------------------end running performance on cross validation---------------------------')
    return np.mean(cv_scores)


def objective4xgb(trial, X, y):
    # 参数网格
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "lambda": trial.suggest_float("lambda", 0, 10),
        "alpha": trial.suggest_float("alpha", 0, 10),
        
        "colsample_bytree": trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]), 
        "colsample_bylevel": trial.suggest_categorical('colsample_bylevel', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        "colsample_bynode": trial.suggest_categorical('colsample_bynode', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        "random_state": 2021,
    }
    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1992)
    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # LGBM建模
        model = xgb.XGBClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
#             callbacks=[
#                 LightGBMPruningCallback(trial, "binary_logloss")
#             ],
        )
        # 模型预测
        preds = model.predict_proba(X_test)[:,1]
        # 优化指标logloss最小, 这里需要不断地去修改, 需要目标函数去改
        thresholds_list = [i * 0.005 for i in range(200)]
        best_threshold, best_score = select_best_threshold(y_test, preds, thresholds_list)
        cv_scores[idx] = best_score
    return np.mean(cv_scores)

def objective4lgb_sinlge(trial, X_trn, y_trn, X_dev, y_dev):
    # 参数网格
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 2000, 5000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "max_depth": trial.suggest_int("max_depth", 10, 20),
#         "min_child_samples": trial.suggest_int("min_child_samples", 10, 1000),
        "alpha": trial.suggest_float("alpha", 0, 0.05),
        "lambda": trial.suggest_float("lambda", 0, 0.05),
#         "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1, step=0.1),
#         "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1, step=0.1),
        "colsample_bytree": trial.suggest_categorical('colsample_bytree', [0.7,0.8,0.9, 1.0]), 
        "colsample_bynode": trial.suggest_categorical('colsample_bynode', [0.7,0.8,0.9, 1.0]),
        "random_state": 2021,
        
    }
    # 5折交叉验证
    model = lgbm.LGBMClassifier(objective="binary", **param_grid)
    model.fit(
        X_trn,
        y_trn,
#             eval_set=[(X_test, y_test)],   # print太多的东西了，麻烦
#             eval_metric="binary_logloss",   # print太多的东西了，麻烦
#             early_stopping_rounds=100,
#             callbacks=[
#                 LightGBMPruningCallback(trial, "binary_logloss")
#             ],
    )
    # 模型预测
    preds_dev = model.predict_proba(X_dev)[:,1]
    # 优化指标logloss最小, 这里需要不断地去修改, 需要目标函数去改
    best_score = roc_auc_score(y_dev, preds_dev)
    print('roc performance: {}'.format(best_score))
    print('---------------------------end running performance---------------------------')
    return best_score
  
  
"""
对于要做cross validation的使用方法
study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
func = lambda trial: objective(trial, X, y)
study.optimize(func, n_trials=20)
"""

"""
对于要使用单模检验的使用方法
study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
func = lambda trial: objective4lgb_sinlge(trial,
                               features2model_trn[[col for col in features2model_trn.columns if col not in ["ID", "LABEL"]]],
                               features2model_trn["LABEL"],
                               features2model_dev[[col for col in features2model_dev.columns if col not in ["ID", "LABEL"]]],
                               features2model_dev["LABEL"], 
                               )
study.optimize(func, n_trials=20)
"""
  
