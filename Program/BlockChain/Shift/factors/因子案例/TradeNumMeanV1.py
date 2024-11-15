"""
仓位管理框架
"""


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df['振幅'] = (df['high'] - df['low']) / df['open']
    df['factor'] = df['振幅'] / df['trade_num']

    df[factor_name] = df['factor'].rolling(n, min_periods=1).mean()

    return df


def signal_multi_params(df, param_list) -> dict:
    volatility = (df['high'] - df['low']) / df['open']
    factor = volatility / df['trade_num'] * 1e5

    ret = dict()
    for param in param_list:
        n = int(param)
        ret[str(param)] = factor.rolling(n, min_periods=1).mean()
    return ret
