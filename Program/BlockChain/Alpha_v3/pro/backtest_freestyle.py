"""
寻找最优参数
"""
import warnings

import pandas as pd

from core.backtest import find_best_params
from core.model.backtest_config import BacktestConfigFactory
from core.utils.log_kit import logger
from core.version import version_prompt

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
    version_prompt()
    logger.info(f'系统启动中，稍等...')

    # ====================================================================================================
    # 1. 配置需要遍历的参数
    # ====================================================================================================
    backtest_name = 'QuoteVolumeMean'
    strategies = [
        # 下面是一个策略的配置，因为我们支持大杂烩，因此定义的是strategy_list，结构上是一个二维数组
        [{
            "strategy": "Strategy_QuoteVolumeMean",
            "offset_list": list(range(0, 1, 1)),
            "hold_period": "1H",
            "is_use_spot": False,
            "cap_weight": 1,
            "factor_list": [
                ('QuoteVolumeMean', True, n, 1)  # 多头因子名（和factors文件中相同），排序方式，参数，权重。
            ],
            "filter_list": [
                ('PctChange', n, 'pct:<0.8')  # 因子名（和factors文件中相同），参数，过滤条件，因子排序方式
            ],
            "use_custom_func": False
        }] for n in list(range(5, 50, 5))
    ]

    factory = BacktestConfigFactory.init(backtest_name=backtest_name)
    factory.generate_configs_by_strategies(strategies=strategies)

    # ====================================================================================================
    # 2. 执行遍历
    # ====================================================================================================
    find_best_params(factory)
