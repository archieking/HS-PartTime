"""
仓位管理框架
"""
import numpy as np
import pandas as pd

from core.model.strategy_config import PosStrategyConfig


def calc_ratio(equity_dfs: [pd.DataFrame], stg_conf: PosStrategyConfig) -> pd.DataFrame:
    """
    计算选币仓位结果，只接受两个参数，1. 资金曲线们，2. 策略配置
    :param equity_dfs: 资金曲线，是一个df的列表，里面包含配置中要求计算好的因子
    :param stg_conf: 策略配置
    :return: 返回仓位比，比如：
                         0    1
    candle_begin_time
    2021-01-01 00:00:00  1.0  0.0
    2021-01-01 06:00:00  1.0  0.0
    2021-01-01 12:00:00  1.0  0.0
    2021-01-01 18:00:00  1.0  0.0
    2021-01-02 00:00:00  1.0  0.0
    ...                  ...  ...
    2024-07-23 06:00:00  0.0  1.0
    2024-07-23 12:00:00  0.0  1.0
    2024-07-23 18:00:00  0.0  1.0

    # 说明：
    可以在这里尽情施展你的组合才华，输入中包含策略的对象，以及我们准备好的资金曲线list
    """
    # 取出对应参数
    cap_ratios = stg_conf.params['cap_ratios']

    # 判断资金占比的参数数量是否与资金曲线的数量一致
    if len(cap_ratios) != len(equity_dfs):
        print('资金比例数量与资金曲线数量不同，退出...')
        exit()

    # 构建时间轴
    begin_times = equity_dfs[0]['candle_begin_time']

    # 创建一个全 0 的 NDArray
    cap_weights = np.zeros([len(begin_times), len(equity_dfs)])

    # 分配权重
    for i in range(len(cap_ratios)):
        cap_weights[:, i] = cap_ratios[i]

    # 将结果转换回 DataFrame
    cap_weights = pd.DataFrame(cap_weights, columns=range(cap_weights.shape[1]), index=begin_times)

    return cap_weights



