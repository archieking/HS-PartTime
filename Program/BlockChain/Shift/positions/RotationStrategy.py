"""
仓位管理框架
"""
import numpy as np
import pandas as pd

from core.model.strategy_config import PosStrategyConfig


def cal_one_ratio(factor_info, equity_dfs: [pd.DataFrame]) -> pd.DataFrame:
    """
    计算选币仓位结果，只接受两个参数，1. 因子配置信息，2. 资金曲线们
    :param factor_info: 因子配置信息
    :param equity_dfs: 资金曲线，是一个df的列表，里面包含配置中要求计算好的因子
    :return: 返回轮动因子计算得到的仓位比，比如：
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
    # 构建时间轴
    begin_times = equity_dfs[0]['candle_begin_time']

    # 取出对应因子名
    factor_name = factor_info[0] + '_' + str(factor_info[2])

    # 生成因子df
    factors = pd.DataFrame(index=begin_times, columns=range(len(equity_dfs)))  # 构建存储因子的df

    # 获取每个策略的因子值
    for idx, df in enumerate(equity_dfs):
        factors[idx] = df[factor_name].values

    # 对因子进行排名
    factor_rank = factors.rank(axis=1, ascending=factor_info[1], method='min')

    # 初始化资金比例rot_ratio
    rot_ratio = np.zeros_like(factor_rank)

    # 找到每行最大值/最小值的出现位置，并设置为1
    for col in factor_rank.columns:
        rot_ratio[factor_rank[col] == factor_rank.min(axis=1), col] = 1

    # 修正多个资金曲线排名相同的情况，当其中某N个资金曲线的排名相同时，平分权重
    ratio_sum = rot_ratio.sum(axis=1)
    for i in range(rot_ratio.shape[1]):
        rot_ratio[:, i] /= ratio_sum

    # 将结果转换回 DataFrame
    rot_ratio = pd.DataFrame(rot_ratio, columns=range(len(equity_dfs)), index=begin_times)

    return rot_ratio


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
    # 取出factor_list
    factor_list = stg_conf.params['factor_list']

    # 处理特殊情况
    if len(factor_list) == 0:
        print('轮动因子为空，退出程序...')
        exit()

    # 构建总轮动资金占比df
    all_rot_ratio = pd.DataFrame()

    # 循环处理每一个轮动因子
    for i, factor_info in enumerate(factor_list):
        print(f'正在处理轮动因子：{factor_info}')

        # 计算轮动因子的资金占比
        rot_ratio = cal_one_ratio(factor_info, equity_dfs)

        # 将该轮动因子的资金占比加到总资金占比中
        all_rot_ratio = rot_ratio.copy() if i == 0 else all_rot_ratio.add(rot_ratio, fill_value=0)

    # 将总轮动资金占比除以覆盖的数量，每行资金占比加总为1
    all_rot_ratio = all_rot_ratio.apply(lambda x: x / len(stg_conf.factor_list))

    return all_rot_ratio
