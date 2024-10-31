"""
选币策略框架
"""
import warnings

import pandas as pd

from core.backtest import run_backtest
from core.model.backtest_config import BacktestConfig
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

    backtest_config = BacktestConfig.init_from_config()
    backtest_config.info()

    run_backtest(backtest_config)
