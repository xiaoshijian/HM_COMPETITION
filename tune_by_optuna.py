from sklearn.metrics import roc_auc_score
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import StratifiedKFold
import optuna
import lightgbm as lgbm
import xgboost as xgb

def select_best_threshold(actual_label, pred_prob, thresholds_list, eval_func=None):
    # 根据模型的表现选择最好的阈值，默认使用auc作为metrics
    if metrics is None:
        eval_func = roc_auc_score
    best_threshold, best_score = None, -1
    for threshold in thresholds_list:
        pred_label = np.where(pred_prob >= threshold, 1, 0)
        score = eval_func(dev_data['LABEL'], pred_label)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    print("best score: {}, corresponding threshold: {}".format(best_score, best_threshold))
    return best_threshold, best_score
  
  
  
def objective4lgbm(trial, X, y):
    # 参数网格
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 100),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 100),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
        "random_state": 2021,
    }
    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1992)
    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # LGBM建模
        model = lgbm.LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "binary_logloss")
            ],
        )
        # 模型预测
        preds = model.predict_proba(X_test)
        # 优化指标logloss最小, 这里需要不断地去修改, 需要目标函数去改
        thresholds_list = [i * 0.005 for i in range(200)]
        best_threshold, best_score = select_best_threshold(y, preds, thresholds_list)
        cv_scores[idx] = best_score
    return np.mean(cv_scores)


def objective4xgb(trial, X, y):
    # 参数网格
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "gamma": trial.suggest_float("learning_rate", 0, 10),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 20, 300)
        "lambda": trial.suggest_float("lambda", 0, 100),
        "alpha": trial.suggest_float("alpha", 0, 100),
        
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
        model = xgb.LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "binary_logloss")
            ],
        )
        # 模型预测
        preds = model.predict_proba(X_test)
        # 优化指标logloss最小, 这里需要不断地去修改, 需要目标函数去改
        thresholds_list = [i * 0.005 for i in range(200)]
        best_threshold, best_score = select_best_threshold(y, preds, thresholds_list)
        cv_scores[idx] = best_score
    return np.mean(cv_scores)


  
  
"""
使用方法
study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
func = lambda trial: objective(trial, X, y)
study.optimize(func, n_trials=20)
"""
  
