import pandas as pd

def assemble_features(features_lists, colname4id):
    # 把多个features拼接起来，每个features都应该有colname4id这个column
    # 是经过特征工程后的方法
    output = None
    for features in features_lists:
        if output is None:
            output = features
        else:
            output = pd.merge(output, features, on=colname4id)
    return output
