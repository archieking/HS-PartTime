"""
选币策略框架
"""
import os
import sys
import warnings

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import pandas as pd
from config import backtest_path, backtest_iter_path
from core.model.backtest_config import BacktestConfig
from core.utils.log_kit import logger
from core.utils.path_kit import get_file_path
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

    # 1. 配置需要聚合的策略
    # ====================================================================================================
    backtest_name_list = ['QuoteVolumeMean']
    # ** 注意 **：目前我们仅针对回测策略的结果进行聚合，可以修改下面循环的规则，对不同的策略进行聚合

    # 2. 遍历data/回测结果目录下的所有策略
    # ====================================================================================================
    report_df_list = []
    for backtest_name in backtest_name_list:
        # 针对回测组和进行聚合
        # all_rtn_files = (backtest_path / backtest_name).rglob(f'策略评价.csv')
        # 针对遍历的各种组合进行聚合
        all_rtn_files = (backtest_iter_path / backtest_name).rglob(f'策略组*/策略评价.csv')

        for rtn_file in all_rtn_files:
            config: BacktestConfig = pd.read_pickle(rtn_file.parent / 'config.pkl')

            backtest_full_name = rtn_file.parts[-2]
            _df = pd.read_csv(rtn_file, encoding='utf-8-sig', index_col=[0]).T
            _df['backtest_name'] = backtest_full_name
            _df['path'] = rtn_file
            # 把策略参数装进去
            for k, v in config.get_strategy_config_sheet(sep_filter=True).items():
                # 排除""#FACTOR-"开头的k
                if k.startswith("#FACTOR-"):  # 因为j神的因子说g_开头的，所以这部分逻辑并不是通用的
                    _df["选币因子"] = k[8:]
                    _df["因子参数"] = v
                elif k.startswith("#FILTER-"):
                    _df["过滤因子"] = k[8:]
                    _df["过滤参数"] = v
                else:
                    _df[k] = v
            report_df_list.append(_df)

    all_rtn_df = pd.concat(report_df_list, ignore_index=True)
    all_rtn_df.to_csv(get_file_path(backtest_path, '策略评价汇总.csv'), encoding='utf-8-sig', index=False)
    print(all_rtn_df)
