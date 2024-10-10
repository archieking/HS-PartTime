"""
自定义一键遍历
"""
import sys
import os

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
from core.utils.log_kit import logger, divider
from core.version import sys_version, build_version
import os
from config import backtest_name
from core.utils.path_kit import get_file_path
import time
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

    divider(f'版本: {sys_version}，当前时间:', '#', _logger=logger)
    logger.debug(f'BUILD VERSION: {build_version}')
    logger.info(f'系统启动中，稍等...')

    factor = 'Pvrsixqq'  # 因子名称
    sl_coin_list = [1, 2]  # 多空选币
    is_use_spot = False  # 是否使用现货
    ascending_list = [False]  # 因子排序
    plot_type = 'Y'
    filter_list = []
    long_filter_list = [
        ('涨跌幅max', 24, 'val:<=0.2'),  # 因子名（和factors文件中相同），参数 rank排名 val数值 pct百分比
        # ('VolumeMix', 24, 'pct:<=0.2', False)  # 因子名（和factors文件中相同），参数
    ]
    short_filter_list = [
        ('涨跌幅max', 24, 'val:<=0.2')  # 因子名（和factors文件中相同），参数
    ]

    strategies = []
    for sl_coin in sl_coin_list:
        for factor_para in [_ for _ in range(10, 201, 10)]:
            for ascending in ascending_list:
                strategy = {
                    "strategy": "Strategy_Base",
                    "offset_list": [0],
                    "hold_period": "1H",
                    "is_use_spot": is_use_spot,
                    "long_select_coin_num": sl_coin,
                    "short_select_coin_num": sl_coin,
                    "cap_weight": 1,
                    "long_cap_weight": 1,  # 策略内多头资金权重
                    "short_cap_weight": 1,  # 空头资金占比
                    "factor_list": [
                        (factor, ascending, factor_para, 1)  # 多头因子名（和factors文件中相同），排序方式，参数，权重。
                    ],
                    "long_filter_list": long_filter_list,
                    "short_filter_list": short_filter_list,
                    "use_custom_func": False  # 使用系统内置因子计算、过滤函数
                }
                strategies.append([strategy])
    #
    factory = BacktestConfigFactory.init(backtest_name=backtest_name)
    factory.generate_configs_by_strategies(strategies=strategies)
    # # # #
    # # # # # ====================================================================================================
    # # # # # 2. 执行遍历
    # # # # # ====================================================================================================
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
        df = pd.read_csv(conf.get_result_folder() / f'{plot_type_file_dict[plot_type]}.csv', encoding='gbk',
                         index_col='candle_begin_time')
        df = df[['累积净值', '收益回撤比', '最大回撤']]
        temp = conf.get_fullname().split(',')
        df.rename(columns={
            '累积净值': f"L{temp[1].split(':')[1]}S{temp[3].split(':')[1]}_" + sheet_df.loc[
                conf.get_fullname(), f'#FACTOR-{factor}'] + '累积净值',
            '收益回撤比': f"L{temp[1].split(':')[1]}S{temp[3].split(':')[1]}_" + sheet_df.loc[
                conf.get_fullname(), f'#FACTOR-{factor}'] + '收益回撤比',
            '最大回撤': f"L{temp[1].split(':')[1]}S{temp[3].split(':')[1]}_" + sheet_df.loc[
                conf.get_fullname(), f'#FACTOR-{factor}'] + '最大回撤',
        }, inplace=True)

        if all_df.empty:
            all_df = df.copy()
        else:
            all_df = all_df.merge(df, left_index=True, right_index=True, how='left')

    all_df.index.name = None
    all_df = all_df.T

    all_df.to_csv(get_file_path('data', '遍历结果', 'temp.csv'))
    print(all_df)

    logger.ok(f'完成参数平原结果输出，花费时间：{time.time() - s_time:.3f}秒')

    # 绘图
    some_folder = list(factory.result_folders)[0]  # 拿出factory中任意一个config使用的folder
    for slc in sl_coin_list:
        draw_df = all_df[all_df.index.str.contains(f'L{slc}S{slc}')]
        draw_df = draw_df.reset_index(drop=False)
        draw_df['index'] = draw_df['index'].str.replace(f'L{slc}S{slc}_', '')
        draw_df.set_index('index', inplace=True)

        filter_name = ''
        if is_use_spot:
            pre = 'Spot'
        else:
            pre = 'Swap'

        if len(filter_list) > 0:
            for f in filter_list:
                filter_name += f"{f}"

        if len(long_filter_list) > 0 :
            filter_name += '_LF'
            for f in long_filter_list:
                filter_name += f"{f}"

        if len(short_filter_list) > 0 :
            filter_name += '_SF'
            for f in short_filter_list:
                filter_name += f"{f}"

        res_name = f"{pre}_{factor}_L{slc}S{slc}{filter_name}"
        draw_df.to_csv(get_file_path('data', '遍历结果', f'{res_name}.csv'))

        plotly_plot(draw_df, some_folder, f'参数平原图_{res_name}')

