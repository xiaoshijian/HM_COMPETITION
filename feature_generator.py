def generate_features_by_operator(array_a, array_b, operator):
    """
    20221020
    对连续性特征进行加减乘除
    """
    assert operator in ["+", "-", "*", "/"]
    output = None
    if operator == "+":
        output = array_a + array_b 
    elif operator == "-":
        output = array_a - array_b
    elif operator == "*":
        output = array_a * array_b
    elif operator == "/":
        output = array_a / array_b
    else:
        pass
    return output

def name_features_by_operator(colname_a, colname_b, operator):
    """
    20221020
    对生成的特征命名
    """
    feature_name = "{} {} {}".format(colname_a, operator, colname_b)
    return feature_name


class FeatureGeneratorByOperator(object):
    def __init__(
            self,
            colnames_for_continous_features,
            colname4id
    ):
        self._colnames_for_continous_features = colnames_for_continous_features[:]
        self._colname4id = colname4id
    
    def generate_features(self, data):  # 或者改为transform更好
        """
        对连续性的特征进行两两结合，并且使用加减乘除生成特征
        """
        features = {
            colname4id: data[self._colname4id],
        }
        colnames_combinations = list(itertools.combinations(self._colnames_for_continous_features, 2))
        elements_combinations = []
        for item in colnames_combinations:
            for operator in ["+", "-", "*", "/"]:
                elements_combinations.append((item[0], item[1], operator))
        
        for colname_a,colname_b, operator in elements_combinations:
            feature_name = name_features_by_operator(colname_a, colname_b, operator)
            feature = generate_features_by_operator(data[colname_a], data[colname_b], operator)
            features[feature_name] = feature
        
        features = pd.DataFrame(features)
        return features
      
 
class FeatureGenerator4CrossProduct(object):
    def __init__(
            sellf,
            colnames_for_cross_product,
            colname4id
    ):
        self._colname4id = colname4id
        self._colnames_for_cross_product = colnames_for_cross_product[:]
    
    def fit(self, data):
        self._colnames_and_ohe_dict = {}  # it is a orderdict
        for colname in self._colnames_for_cross_product:
            ohe = OneHotEncoder()
            self._colnames_and_ohe_dict[colname] = ohe.fit(
                {colname: data[colname].astype("category")}
            )
    
    def transform(self, data):
        n = len(self._colnames_for_cross_product)
        features = {self._colname4id: data[self._colname4id]}
        for i in range(n):
            for j in range(i+1, n):
                first_feature_for_cross_product = self._colnames_for_cross_product[i]
                second_feature_for_cross_product = self._colnames_for_cross_product[j]
                ohe_for_first_feature = self._colnames_and_ohe_dict[first_feature_for_cross_product]
                ohe_for_first_feature = self._colnames_and_ohe_dict[first_feature_for_cross_product]
                a =  ohe_for_first_feature.transform(
                           pd.DataFrame(
                               {first_feature_for_cross_product: data[first_feature_for_cross_product].astype("category")})
                )
                b =  ohe_for_first_feature.transform(
                           pd.DataFrame(
                               {second_feature_for_cross_product: data[second_feature_for_cross_product].astype("category")})
                )
                for colname_a in a.columns:
                    for colname_b in b.columns:
                        new_feature_name = "({})_({})".format(colname_a, colname_b)
                        new_feature = a[colname_a] * b[colname_b]
                        features[new_feature_name] = new_feature
        features = pd.DataFrame(features)
        return features
  
