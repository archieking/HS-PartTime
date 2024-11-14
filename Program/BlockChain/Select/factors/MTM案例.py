#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 名字：MTM案例
# 选币：1,2
# 范围：56-67
# {
#         "strategy": "Strategy_Base",
#         "offset_list": [0],
#         "hold_period": "1H",
#         "is_use_spot": True,
#         "long_select_coin_num": 1,
#         "short_select_coin_num": 1,
#         "cap_weight": 1,
#         "long_cap_weight": 1,  # 多头资金占比
#         "short_cap_weight": 1,  # 空头资金占比
#         "factor_list": [
#             ('MTM案例', False, 100, 1)  # 因子名（和factors文件中相同），排序方式，参数，权重。
#         ],
#         "long_filter_list": [
#         ],
#         "short_filter_list": [
#         ],
#         "use_custom_func": False  # 使用系统内置因子计算、过滤函数
#     }

def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df['mtm'] = df['close'] / df['close'].shift(n) - 1
    df[factor_name] = df['mtm'].rolling(window=n, min_periods=1).mean()

    return df