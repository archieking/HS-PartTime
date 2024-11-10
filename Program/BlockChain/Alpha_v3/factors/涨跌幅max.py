#!/usr/bin/python3
# -*- coding: utf-8 -*-


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df['该小时涨跌幅'] = abs(df['close'].pct_change(1))
    df[factor_name] = df['该小时涨跌幅'].rolling(n, min_periods=1).max()

    return df
