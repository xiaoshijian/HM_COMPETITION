
"""
暂时用于存放计算信息指标的程序
"""
import numpy as np

def cal_iv4catfeature(feature, target):
    """
    用于计算categorical字段的information value
    information value的计算公式可以参考晚上的资料
    #TO DO LIST：对于少于一定ratio的类别组合成多个OTHERS
    """
    df = pd.DataFrame({"feature": feature, "target": target})
    n = len(df)
    ng = sum(target == 1)
    nb = sum(target == 0)
    a = df.groupby("feature")["target"].apply(
        lambda s: (sum(s == 0)/nb - sum(s == 1)/ng) * np.log( (sum(s == 0)/nb + 0.0001) / (sum(s == 1)/ng + 0.0001)) 
    ).reset_index()
    output = sum(a["target"])
    return output
  
  def cal_mutualinfo4crossproduct(column_a, column_b):
    """
    用于计算俩列的数据（indicator数据）的互信息
    column_a: np.array, array of indicator
    column_b: np.array, array of indicator
    return: mutual information
    """
    n = len(column_a)
    output = np.log( sum(column_a * column_b) * n / (sum(column_a) * sum(column_b)) + 0.00001)
    return output
