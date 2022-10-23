from tying import List
from .info_index import cal_mutualinfo4crossproduct

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
        
    def fit(self, data):
        # 就是一个dummy的接口
        return 
    
    def transform(self, data):  # 或者改为transform更好
        """
        对连续性的特征进行两两结合，并且使用加减乘除生成特征
        """
        features = {
            colname4id: data[self._colname4id],
        }
        colnames_combinations = list(itertools.combinations(self._colnames_for_continous_features, 2))
        elements_combinations = []
        for item in colnames_combinations:
            for operator in ["+", "*", "/"]:  
            # for operator in ["+", "-", "*", "/"]:  # TODO，在选特征的时候，遇到nonnegative的问题了，暂时就不用减少的了，以后解决这屁问题再添加
                
                elements_combinations.append((item[0], item[1], operator))
        
        for colname_a,colname_b, operator in elements_combinations:
            feature_name = name_features_by_operator(colname_a, colname_b, operator)
            feature = generate_features_by_operator(data[colname_a], data[colname_b], operator)
            features[feature_name] = feature
        
        features = pd.DataFrame(features)
        return features

      
 
class FeatureGenerator4CrossProduct(object):
    def __init__(
            self,
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
                pd.DataFrame({colname: data[colname].astype("category")}) 
            )
    
    def transform(self, data):
        n = len(self._colnames_for_cross_product)
        features = {self._colname4id: data[self._colname4id]}
        for i in range(n-1):
            for j in range(i+1, n):
                first_feature_for_cross_product = self._colnames_for_cross_product[i]
                second_feature_for_cross_product = self._colnames_for_cross_product[j]
                ohe_for_first_feature = self._colnames_and_ohe_dict[first_feature_for_cross_product]
                ohe_for_second_feature = self._colnames_and_ohe_dict[second_feature_for_cross_product]
                a =  ohe_for_first_feature.transform(
                           pd.DataFrame(
                               {first_feature_for_cross_product: data[first_feature_for_cross_product].astype("category")})
                )
                b =  ohe_for_second_feature.transform(
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
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    
class FeatureGenerator4CatFeatureWOE(object):
    def __init__(
            self,
            colnames_for_woe_features,
            colname4id,
            colname4label
    ):
        self._colname4id = colname4id
        self._colnames_for_woe_features = colnames_for_woe_features[:]
        self._colname4label = colname4label
        
    def fit(self, data):
        self._woeencoder = WOEEncoder()  # it is a orderdict
        self._woeencoder.fit(data[self._colnames_for_woe_features].astype("category"), data[self._colname4label])
        return
    
    def transform(self, data):
        woe_features = self._woeencoder.transform(data[self._colnames_for_woe_features].astype("category"))
        woe_features.columns = [str(col) + '_WOE' for col in woe_features.columns]
        woe_features[self._colname4id] = data[self._colname4id]
        return woe_features
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    

class FeatureGenerator4CrossProductByMutualInfo(object):
    def __init__(
            self,
            colnames_for_cross_product: List,
            colname4id: str,
            colname4label: str,
            ratio=0.01
    ):
        """
        ratio: 根据互信息提取的cross_product的比例，默认1%
        """
        self._colname4id = colname4id
        self._colnames_for_cross_product = colnames_for_cross_product[:]
        self._ratio = ratio
    
    def fit(self, data):
        n = len(self._colnames_for_cross_product)
        self._colnames_and_ohe_dict = {}  # it is a orderdict
        for colname in self._colnames_for_cross_product:
            ohe = OneHotEncoder()
            self._colnames_and_ohe_dict[colname] = ohe.fit(
                pd.DataFrame({colname: data[colname].astype("category")}) 
            )
            
        # 先判断每个组合的名称以及互信息
        # TODO（xiaoshijian）：可以添加更多的筛选条件, 可能可以用矩阵优化
        tmp_list = []
        for i in range(n-1):
            for j in range(i+1, n):
                first_feature_for_cross_product = self._colnames_for_cross_product[i]
                second_feature_for_cross_product = self._colnames_for_cross_product[j]
                ohe_for_first_feature = self._colnames_and_ohe_dict[first_feature_for_cross_product]
                ohe_for_second_feature = self._colnames_and_ohe_dict[second_feature_for_cross_product]
                a =  ohe_for_first_feature.transform(
                           pd.DataFrame(
                               {first_feature_for_cross_product: data[first_feature_for_cross_product].astype("category")})
                )
                b =  ohe_for_second_feature.transform(
                           pd.DataFrame(
                               {second_feature_for_cross_product: data[second_feature_for_cross_product].astype("category")})
                )
                #TODO: 用矩阵的方法应该会快很多的，(len(a.columns) * n) dot (n * len(b.columns)) 
                for colname_a in a.columns:
                    for colname_b in b.columns:
                        new_feature_name = "({})_({})".format(colname_a, colname_b)
                        mutual_info_for_new_feature = cal_mutualinfo4crossproduct(a[colname_a], b[colname_b]) 
                        tmp_list.append([new_feature_name, mutual_info_for_new_feature])
        # 对生成的特征进行排序并且提取
        m = len(tmp_list)
        tmp_list = sorted(tmp_list, key = lambda x: x[1], reverse = True)
        tmp_list = tmp_list[:int(self._ratio * m)]
        
        # 把挑选后的特征以字典的方式保存下来
        self._selected_cross_product_features = {x[0]: x[1] for x in tmp_list}
                
    
    def transform(self, data):
        n = len(self._colnames_for_cross_product)
        features = {self._colname4id: data[self._colname4id]}
        for i in range(n-1):
            for j in range(i+1, n):
                first_feature_for_cross_product = self._colnames_for_cross_product[i]
                second_feature_for_cross_product = self._colnames_for_cross_product[j]
                ohe_for_first_feature = self._colnames_and_ohe_dict[first_feature_for_cross_product]
                ohe_for_second_feature = self._colnames_and_ohe_dict[second_feature_for_cross_product]
                a =  ohe_for_first_feature.transform(
                           pd.DataFrame(
                               {first_feature_for_cross_product: data[first_feature_for_cross_product].astype("category")})
                )
                b =  ohe_for_second_feature.transform(
                           pd.DataFrame(
                               {second_feature_for_cross_product: data[second_feature_for_cross_product].astype("category")})
                )
                for colname_a in a.columns:
                    for colname_b in b.columns:
                        new_feature_name = "({})_({})".format(colname_a, colname_b)
                        if new_feature_name not in self._selected_cross_product_features:
                            continue
                        new_feature = a[colname_a] * b[colname_b]
                        features[new_feature_name] = new_feature
        features = pd.DataFrame(features)
        return features
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

