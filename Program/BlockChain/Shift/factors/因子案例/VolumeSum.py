"""
仓位管理框架
"""

eps = 1e-8


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df[factor_name] = df['quote_volume'].rolling(n, min_periods=1).sum()

    return df


def get_parameter():
    param_list = []
    n_list = [24]
    for n in n_list:
        param_list.append(n)

    return param_list