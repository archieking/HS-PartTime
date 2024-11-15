"""
仓位管理框架
"""


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df['MtmMean'] = df['close'].pct_change(n).rolling(n).mean()
    df['std'] =  df['close'].pct_change(n).rolling(n).std()
    df[factor_name] = (df['MtmMean'] * df['std'])

    return df
