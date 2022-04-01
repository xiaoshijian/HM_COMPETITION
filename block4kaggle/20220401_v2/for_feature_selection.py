import xgboost as xgb
import lightgbm as lgb
import numpy as np
import numpy


class mvtest:
    def __init__(self):
        self._n = None
        self._m = None
        self.quantiles = [
            [0.1178, 0.2091, 0.2442, 0.2860, 0.3469, 0.4665, 0.4968, 0.5416, 0.5984, 0.7120],
            [0.2773, 0.4227, 0.4631, 0.5220, 0.6086, 0.7365, 0.7776, 0.8377, 0.9278, 1.0673],
            [0.4385, 0.6231, 0.6761, 0.7495, 0.8384, 0.9938, 1.0453, 1.1055, 1.2090, 1.3393],
            [0.6028, 0.8158, 0.8737, 0.9519, 1.0602, 1.2399, 1.2911, 1.3622, 1.4408, 1.6026],
            [0.7694, 1.0097, 1.0786, 1.1670, 1.2698, 1.4440, 1.4955, 1.5776, 1.6992, 1.8650],
            [0.9423, 1.2035, 1.2757, 1.3627, 1.4927, 1.6986, 1.7672, 1.8268, 1.9691, 2.1241],
            [1.0971, 1.3787, 1.4569, 1.5549, 1.6780, 1.8901, 1.9581, 2.0594, 2.1693, 2.3645],
            [1.2708, 1.5774, 1.6674, 1.7672, 1.8963, 2.1096, 2.1742, 2.2597, 2.3809, 2.6180],
            [1.4468, 1.7812, 1.8707, 1.9773, 2.1019, 2.3362, 2.4081, 2.5011, 2.6215, 2.7943]
        ]
        self.prob = [0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99]

    def _f(self, s, x):
        return len(x[x <= s]) / self._n

    def _fr(self, s, t, x, y):
        return len(x[(x <= s) & (y == t)]) / len(y[y == t])

    def _pr(self, t, y):
        return len(y[y == t]) / self._n

    def _quantiles_transformer(self, result):
        quantiles = self.quantiles[self._m - 2]
        if result < quantiles[0]:
            return [0.50, 1]
        if quantiles[len(quantiles) - 1] <= result:
            return [0, 0.01]
        for i in range(len(quantiles) - 1):
            if quantiles[i] <= result < quantiles[i + 1]:
                return [round(1 - self.prob[i + 1], 2), round(1 - self.prob[i], 2)]

    def test(self, x: numpy.ndarray, y: numpy.ndarray) -> dict:

        try:
            x = numpy.array(x)
        except Exception:
            raise Exception("The expected type of this argument is Array or List,"
                            " however \"" + str(type(x)) + "\" is gotten.")
        try:
            y = numpy.array(y, dtype=int)
        except Exception:
            raise Exception("The expected type of this argument is Array or List,"
                            " however \"" + str(type(y)) + "\" is gotten.")
        if len(x) == 0 or len(y) == 0:
            raise Exception("The input vectors cannot be empty.")
        if type(x.dtype) == str or type(y.dtype) == str:
            raise Exception("The element type of input vectors cannot be \"str\".")

        self._n = len(x)
        self._m = len(numpy.unique(y))
        if self._m > 10:
            self._m = 10

        if self._n == len(y):
            result = 0
            for t in numpy.unique(y):
                pr = self._pr(t, y)
                for s in x:
                    result += pr * numpy.square(self._fr(s, t, x, y) - self._f(s, x))

            return {'Tn': round(result, 2), 'p-value': self._quantiles_transformer(result)}
        else:
            raise Exception("Two vectors must be equal to the same dimension vector.")

    def test_accelerate(self, x: numpy.ndarray, y: numpy.ndarray):
        try:
            x = numpy.array(x)
        except Exception:
            raise Exception("The expected type of this argument is Array or List,"
                            " however \"" + str(type(x)) + "\" is gotten.")
        try:
            y = numpy.array(y, dtype=int)
        except Exception:
            raise Exception("The expected type of this argument is Array or List,"
                            " however \"" + str(type(y)) + "\" is gotten.")
        if len(x) == 0 or len(y) == 0:
            raise Exception("The input vectors cannot be empty.")
        if type(x.dtype) == str or type(y.dtype) == str:
            raise Exception("The element type of input vectors cannot be \"str\".")

        self._n = len(x)
        self._m = len(numpy.unique(y))
        if self._m > 10:
            self._m = 10

        if self._n == len(y):
            result = 0
            xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=-1)
            values_info_in_xy = get_values_info(xy)
            for t in np.unique(y):
                xyr = xy[xy[:, 1] == t]
                pr = len(xyr) / len(xy)
                values_info_in_xyr = get_values_info(xyr)
                partial_result = cal_partial_result(pr, values_info_in_xy, values_info_in_xyr)
                result += partial_result

            return {'Tn': round(result, 2), 'p-value': self._quantiles_transformer(result)}
        else:
            raise Exception("Two vectors must be equal to the same dimension vector.")


def get_values_info(xy):
    '''
    get information for each value in xy
    information include: (value,
                          last position of value in ratio,
                          last position of value ,
                          total amount of value)
    '''
    xy = xy[xy[:, 0].argsort()]
    x = xy[:, 0]
    size = len(x)
    value_set = set()
    output = []
    for i in range(len(x) - 1, -1, -1):
        value = x[i]
        if value not in value_set:
            value_set.add(value)
            # (value, last position ratio of value in list,  last position of value in list)
            output.append([value, (i + 1) / size, i + 1])
    output = output[::-1]  # reverse it

    for i in range(len(output)):
        # add total amount of this value in list to item
        if i == 0:
            output[i].append(output[i][2])
        # last position in current value - last position in last value, because it is sorted
        else:
            output[i].append(output[i][2] - output[i - 1][2])
    return output


def cal_partial_result(pr, values_info_in_xy, values_info_in_xyr):
    # x_in_xy, x_in_xyr均为list，其中的元素是（value，position_in_percentage）
    l1, l2 = len(values_info_in_xy), len(values_info_in_xyr)
    p1, p2 = 0, 0
    partial_result = 0

    while p2 < l2 and p1 < l1:
        # 对于x_in_xy中的每一个元素，寻找在它在x_in_xyr的位置，并找出它的前一个元素
        s = values_info_in_xy[p1][0]
        ratio1 = values_info_in_xy[p1][1]
        value_amount_in_xy = values_info_in_xy[p1][3]

        if s >= values_info_in_xyr[p2][0]:
            while p2 < l2 and s >= values_info_in_xyr[p2][0]:
                p2 += 1
            ratio2 = values_info_in_xyr[p2 - 1][1]
        else:
            if p2 == 0:
                ratio2 = 0
            else:
                ratio2 = values_info_in_xyr[p2 - 1][1]
        partial_result += value_amount_in_xy * pr * np.square(ratio2 - ratio1)
        p1 += 1

    for p in range(p1, l1):
        ratio1 = values_info_in_xy[p1][1]
        value_amount_in_xy = values_info_in_xy[p1][2]
        ratio2 = 1
        partial_result += value_amount_in_xy * pr * np.square(ratio2 - ratio1)
    return partial_result


def get_mv_for_features(data, customer_id, label):
    model = mvtest()
    output = {}
    for col in data.columns:
        if col in [customer_id, label]:
            continue
        try:
            mv4feature = model.test_accelerate(data[col], data[label])
            output[col] = mv4feature
        except Exception as e:
            print(col)
            print(e)
            pass
    return output


def get_importance_from_ranker(data, customer_id, label, paras=None, mode='lgb'):
    # use the api "feature_importances_" in tree model (skilearn)
    # to get feature importance
    data.sort_values(by=[customer_id], inplace=True)
    g_data = data.groupby([customer_id], as_index=False).count()[label].values

    columns_in_x = [col for col in data.columns if col not in [customer_id, label]]

    if mode == 'lgb':
        if paras is None:
            paras = {
                'n_estimators': 1000,
            }
        ranker = lgb.LGBMRanker(boosting_type='gbdt')

    elif mode == 'xgb':
        if paras is None:
            paras = {
                'objective': 'binary:logistic',
                'n_estimators': 1000,
            }
        clf = xgb.XGBClassifier(**paras)

    ranker.fit(data[columns_in_x], data[[label]], group=g_data)
    feature_importance = ranker.feature_importances_
    feature_and_importance = [(col, value) for col, value in zip(columns_in_x, feature_importance)]
    return (ranker, feature_importance, feature_and_importance)


'''
above is code
---------------------------------------------------------
below is running code
'''

mv_for_features = get_mv_for_features(
    data=samples_feature4finedrank_trn_cv,
    customer_id='customer_id',
    label='label',
)

selected_features_from_mv = []
for feature, info in mv_for_features.items():
    if info['p-value'][1] <= 0.05:
        selected_features_from_mv.append(feature)

print('feature from mv test: ')
print(selected_features_from_mv)

_, feature_importance_from_lgb, features_and_importance_from_lgb = get_importance_from_ranker(
    data=samples_feature4finedrank_trn_cv,
    customer_id='customer_id',
    label='label',
    mode='lgb',
)

features_name = [col for col in samples_feature4finedrank_trn_cv.columns if col != 'label']
features_and_importance_from_lgb = sorted(features_and_importance_from_lgb, key=lambda x: x[1], reverse=True)
selected_features_from_lgb = [
    item[0] for item in features_and_importance_from_lgb[:int(0.6 * len(features_and_importance_from_lgb))]
]

print('feature from lgb: ')
print(selected_features_from_lgb)

selected_features = list(set(selected_features_from_mv) | set(selected_features_from_lgb))

for item in ['FN', 'Active']:
    if item not in selected_features:
        selected_features.append(item)