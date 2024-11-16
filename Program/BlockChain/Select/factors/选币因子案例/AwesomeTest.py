# -*- coding: utf-8 -*-
"""
选币策略框架
"""


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df['tp'] = (df['close'] + df['high'] + df['low'] + df['open']) / 4
    df['c_bias'] = df['tp'] / df['tp'].rolling(n, min_periods=1).mean()

    df['qv_bias'] = df['quote_volume'] / df['quote_volume'].rolling(n, min_periods=1).mean()
    df['v_bias'] = df['volume'] / df['volume'].rolling(n, min_periods=1).mean()
    df['trade_num'] = df['trade_num'] / df['trade_num'].rolling(n, min_periods=1).mean()
    df['tbv_bias'] = df['taker_buy_base_asset_volume'] / df['taker_buy_base_asset_volume'].rolling(n, min_periods=1).mean()
    df['tbqv_bias'] = df['taker_buy_quote_asset_volume'] / df['taker_buy_quote_asset_volume'].rolling(n, min_periods=1).mean()

    df[factor_name] = df['c_bias'] * df['qv_bias'] * df['v_bias'] * df['trade_num'] * df['tbv_bias'] * df['tbqv_bias']

    return df
