"""
仓位管理框架
"""
import numpy as np


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df[factor_name] = np.where(df['symbol'] == 'BTC-USDT', 1, np.nan)

    return df
