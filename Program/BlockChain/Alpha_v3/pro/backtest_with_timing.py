"""
选币策略框架
"""
import time
import warnings

import pandas as pd

from config import backtest_path, raw_data_path
from core.backtest import run_backtest
from core.equity import calc_equity, show_plot_performance
from core.model.backtest_config import BacktestConfig
from core.model.timing_signal import TimingSignal
from core.select_coin import (agg_multi_strategy_ratio, calc_factors, concat_select_results, process_select_results,
                              select_coins)
from core.utils.functions import load_spot_and_swap_data, save_performance_df_csv
from core.utils.log_kit import divider, logger
from core.utils.path_kit import get_file_path
from core.version import build_version, sys_version

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
    divider(f'版本: {sys_version}，当前时间:', '#', _logger=logger)
    logger.debug(f'BUILD VERSION: {build_version}')
    logger.info(f'系统启动中，稍等...')

    backtest_config = BacktestConfig.init_from_config()
    backtest_config.info()

    run_backtest(backtest_config)
