# -*- coding: utf-8 -*-
"""
选币策略框架
"""


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['std'] = df['close'].rolling(n, min_periods=1).std()
    df['zscore'] = (df['close'] - df['ma']) / df['std']
    df['zscore'] = df['zscore'].abs().rolling(n, min_periods=1).mean().shift()
    df['down'] = df['ma'] - df['zscore'] * df['std']

    df['diff'] = df['close'] - df['down']
    df['min'] = df['diff'].rolling(n, min_periods=1).min()
    df['max'] = df['diff'].rolling(n, min_periods=1).max()

    df[factor_name] = (df['diff'] - df['min']) / (df['max'] - df['min'])

    return df

