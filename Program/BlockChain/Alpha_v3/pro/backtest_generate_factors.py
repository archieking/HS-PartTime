"""
寻找最优参数
"""
import os
import sys

from core.backtest import find_best_params

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import warnings
import pandas as pd
from core.model.backtest_config import BacktestConfigFactory
from core.utils.log_kit import logger
from core.version import version_prompt
import os

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
    backtest_name = 'IterFactors'
    factor_list = []
    # factor_list 是factors文件夹下，所有"g_"开头的脚本名，去掉".py"
    factors_dir = os.path.join(os.path.dirname(__file__), 'factors')
    for filename in os.listdir(factors_dir):
        if filename.startswith('g_') and filename.endswith('.py'):
            factor_name = filename[:-3]  # Remove '.py' extension
            factor_list.append(factor_name)
    print(f"Found factors: {len(factor_list)}")

    strategies = []
    for factor in factor_list:
        for n in [200, 500]:
            for ascending in [True, False]:
                strategy = {
                    "strategy": "Strategy_Test",
                    "offset_list": list(range(0, 24, 1)),
                    "hold_period": "24H",
                    "is_use_spot": False,
                    "cap_weight": 1,
                    "long_cap_weight": 1,  # 策略内多头资金权重
                    "short_cap_weight": 0,  # 策略内空头资金权重
                    "factor_list": [
                        (factor, ascending, n, 1)  # 多头因子名（和factors文件中相同），排序方式，参数，权重。
                    ],
                    "filter_list": [
                        ('PctChange', n, 'pct:<0.8')  # 因子名（和factors文件中相同），参数
                    ],
                    "use_custom_func": False
                }
                strategies.append([strategy])

    factory = BacktestConfigFactory.init(backtest_name=backtest_name)
    factory.generate_configs_by_strategies(strategies=strategies)

    # ====================================================================================================
    # 2. 执行遍历
    # ====================================================================================================
    find_best_params(factory)
