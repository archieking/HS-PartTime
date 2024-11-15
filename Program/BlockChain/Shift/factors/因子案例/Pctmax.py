"""
仓位管理框架
"""


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]
    df['该小时涨跌幅'] = abs(df['close'].pct_change(1))
    df[factor_name] = df['该小时涨跌幅'].rolling(n).max()

    return df


def get_parameter():
    param_list = []

    n_list = [24]

    for n in n_list:
        param_list.append(n)

    return param_list
