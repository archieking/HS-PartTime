"""
寻找最优参数
"""
import warnings

import pandas as pd

from core.backtest import find_best_params
from core.model.backtest_config import BacktestConfigFactory
from core.utils.log_kit import logger
from core.version import version_prompt
import time
from core.utils.path_kit import get_file_path
from core.figure import plotly_plot

# ====================================================================================================
# ** 脚本运行前配置 **
# 主要是解决各种各样奇怪的问题们
# ====================================================================================================
# region 脚本运行前准备
warnings.filterwarnings('ignore')  # 过滤一下warnings，不要吓到老实人

# pandas相关的显示设置，基础课程都有介绍
pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.unicode.ambiguous_as_wide', True)  # 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.east_asian_width', True)

if __name__ == '__main__':
    s_time = time.time()

    version_prompt()
    logger.info(f'系统启动中，稍等...')

    plot_type = 'Y'
    factor = '只选BTC'  # 选币因子
    factor_timing = 'MovingAverage'  # 择时因子
    factor_timing_para_list = range(10, 501, 10)  # 择时因子遍历范围

    # ====================================================================================================
    # 1. 配置需要遍历的参数
    # ====================================================================================================
    strategies = [
        # 下面是一个策略的配置，因为我们支持大杂烩，因此定义的是strategy_list，结构上是一个二维数组
        [{
            "strategy": "Strategy_Base",
            "offset_list": list(range(0, 1, 1)),
            "hold_period": "1H",
            "is_use_spot": True,
            "long_select_coin_num": 1,
            "short_select_coin_num": 0,
            "cap_weight": 1,
            "long_cap_weight": 1,  # 多头资金占比
            "short_cap_weight": 0,  # 空头资金占比
            "factor_list": [
                (factor, True, 100, 1)  # 多头因子名（和factors文件中相同），排序方式，参数，权重。
            ],
            "use_custom_func": False  # 使用系统内置因子计算、过滤函数
        }]
    ]

    # re_timing_strategies = None # 没有再择时则为 None
    re_timing_strategies = [{'name': factor_timing, 'params': [n]} for n in factor_timing_para_list]

    factory = BacktestConfigFactory.init(backtest_name='择时遍历')
    factory.generate_configs_by_strategies(strategies=strategies, re_timing_strategies=re_timing_strategies)

    # ====================================================================================================
    # 2. 执行遍历
    # ====================================================================================================
    find_best_params(factory)

    # 3. 准备参数平原结果输出
    # ====================================================================================================
    # 获取策略配置 -> 参数 的数据图
    sheet_df = factory.get_name_params_sheet()
    sheet_df.set_index('fullname', inplace=True)

    plot_type_file_dict = {
        'Y': '年度账户收益',
        'Q': '季度账户收益',
        'M': '月度账户收益',
    }

    # 读取所有参数的历年表现结果
    all_df = pd.DataFrame()
    for conf in factory.config_list:
        df = pd.read_csv(conf.get_result_folder() / f'{plot_type_file_dict[plot_type]}_再择时.csv', encoding='utf-8-sig',
                         index_col='candle_begin_time')
        # conf.get_fullname().split(',')[1]conf.get_fullname().split(',')[3]
        df = df[['累积净值', '收益回撤比', '最大回撤']]
        # temp = conf.get_fullname().split('再择时:')
        temp = conf.get_fullname().split('再择时:')[1].split('[')[1][:-2]
        df.rename(columns={
            '累积净值': f'{temp}累积净值',
            '收益回撤比': f'{temp}收益回撤比',
            '最大回撤': f'{temp}最大回撤',

        }, inplace=True)

        if all_df.empty:
            all_df = df.copy()
        else:
            all_df = all_df.merge(df, left_index=True, right_index=True, how='left')

    all_df.index.name = None
    all_df = all_df.T

    # all_df.to_csv(get_file_path('data', '遍历结果', 'temp.csv'))
    print(all_df)

    logger.ok(f'完成参数平原结果输出，花费时间：{time.time() - s_time:.3f}秒')

    # 绘图
    res_name = f"{factor}_{factor_timing}"
    all_df.to_csv(get_file_path('data', '遍历结果', f'{res_name}.csv'))

    plotly_plot(all_df, factory.result_folder, f'择时策略_{res_name}')