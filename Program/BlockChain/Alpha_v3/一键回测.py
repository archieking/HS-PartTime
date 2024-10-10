"""
选币策略框架
"""
import time
import warnings

import pandas as pd

from config import raw_data_path
from core.backtest import step2_load_data, step3_calc_factors, step4_select_coins, step5_aggregate_select_results
from core.equity import calc_equity, show_plot_performance
from core.model.backtest_config import BacktestConfig
from core.utils.functions import save_performance_df_csv
from core.utils.log_kit import logger, divider
from core.version import sys_version, build_version

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


# ====================================================================================================
# ** 回测主程序 **
# 1. 准备工作
# 2. 读取数据
# 3. 计算因子
# 4. 选币
# 5. 整理选币数据
# 6. 添加下一个每一个周期需要卖出的币的信息
# 7. 计算资金曲线
# ====================================================================================================
def run_backtest(conf: BacktestConfig):
    # ====================================================================================================
    # 1. 准备工作
    # ====================================================================================================
    divider(conf.name, '*')

    # 记录一下时间戳
    r_time = time.time()

    # 缓存当前的config
    conf.save()

    # ====================================================================================================
    # 2. 读取回测所需数据，并做简单的预处理
    # ====================================================================================================
    step2_load_data(conf)

    # ====================================================================================================
    # 3. 计算因子
    # ====================================================================================================
    step3_calc_factors(conf)

    # ====================================================================================================
    # 4. 选币
    # - 注意：选完之后，每一个策略的选币结果会被保存到硬盘
    # ====================================================================================================
    step4_select_coins(conf)

    # ====================================================================================================
    # 5. 整理选币结果并形成目标持仓
    # ====================================================================================================
    df_spot_ratio, df_swap_ratio = step5_aggregate_select_results(conf)
    logger.ok(f'目标持仓信号已完成，花费时间：{(time.time() - r_time):.3f}秒')

    # ====================================================================================================
    # 6. 根据目标持仓计算资金曲线
    # ====================================================================================================
    logger.info(f'开始模拟交易，累计回溯 {len(df_spot_ratio):,} 小时（~{len(df_spot_ratio) / 24:,.0f}天）...')
    pivot_dict_spot = pd.read_pickle(raw_data_path / 'market_pivot_spot.pkl')
    pivot_dict_swap = pd.read_pickle(raw_data_path / 'market_pivot_swap.pkl')
    account_df, rtn, year_return, month_return, quarter_return = calc_equity(conf, pivot_dict_spot, pivot_dict_swap,
                                                                             df_spot_ratio, df_swap_ratio)
    save_performance_df_csv(conf,
                            资金曲线=account_df,
                            策略评价=rtn,
                            年度账户收益=year_return,
                            季度账户收益=quarter_return,
                            月度账户收益=month_return)

    show_plot_performance(conf, account_df, rtn, year_return)

    logger.ok(f'完成，回测时间：{time.time() - r_time:.3f}秒')


if __name__ == '__main__':
    divider(f'版本: {sys_version}，当前时间:', '#', _logger=logger)
    logger.debug(f'BUILD VERSION: {build_version}')
    logger.info(f'系统启动中，稍等...')

    backtest_config = BacktestConfig.init_from_config()
    backtest_config.info()

    run_backtest(backtest_config)
