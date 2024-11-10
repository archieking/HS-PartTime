# -*- coding: utf-8 -*-
"""
选币策略框架
"""


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df['tp'] = (df['close'] + df['high'] + df['low']) / 3
    df['tp2'] = df['tp'].ewm(span=n, adjust=False).mean()
    df['diff'] = df['tp'] - df['tp2']
    df['min'] = df['diff'].rolling(n, min_periods=1).min()
    df['max'] = df['diff'].rolling(n, min_periods=1).max()

    df[factor_name] = (df['diff'] - df['min']) / (df['max'] - df['min'])

    return df

