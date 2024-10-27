"""
寻找最优参数
"""
import time
import warnings

import pandas as pd

from core.backtest import find_best_params
from core.figure import plotly_plot
from core.model.backtest_config import BacktestConfigFactory
from core.utils.log_kit import logger, divider
from core.version import sys_version, build_version

from config import factor_param_range_dict, strategy_list, params_plot_type
from core.utils.path_kit import get_file_path
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


def run(factor, plot_type='Y'):
    s_time = time.time()
    plot_type_file_dict = {
        'Y': '年度账户收益',
        'Q': '季度账户收益',
        'M': '月度账户收益',
    }
    # ====================================================================================================
    # 1. 配置需要遍历的参数
    # ====================================================================================================
    logger.warning(f'配置遍历的因子：{factor}')

    # ====================================================================================================
    # 2. 生成策略配置
    # ====================================================================================================
    logger.info(f'生成策略配置...')
    backtest_factory = BacktestConfigFactory.init(factor_param_range_dict=factor_param_range_dict)
    backtest_factory.generate_configs_by_factor(factor)

    # 检查配置是否有问题
    if factor not in map(lambda x: x[0], backtest_factory.factor_list | backtest_factory.filter_list):
        logger.error('配置不匹配，目前策略如下：')
        for backtest_config in backtest_factory.config_list:
            logger.debug(f'- {backtest_config.get_fullname()}')
        logger.critical(f'策略并没有使用目标因子：{factor}')
        logger.info(f'请检查 `config.py` 中的策略，和现在需要循环的factor({factor})是否匹配')
        exit(1)

    # ====================================================================================================
    # 3. 寻找最优参数
    # ====================================================================================================
    find_best_params(backtest_factory)

    # ====================================================================================================
    # 4. 准备参数平原结果输出
    # ====================================================================================================
    # 获取策略配置 -> 参数 的数据图
    sheet_df = backtest_factory.get_name_params_sheet()
    sheet_df.set_index('fullname', inplace=True)

    # 读取所有参数的历年表现结果
    all_df = pd.DataFrame()
    for conf in backtest_factory.config_list:
        df = pd.read_csv(conf.get_result_folder() / f'{plot_type_file_dict[plot_type]}.csv', encoding='gbk',
                         index_col='candle_begin_time')
        # df.rename(columns={'涨跌幅': sheet_df.loc[conf.get_fullname(), f'#FACTOR-{factor}']}, inplace=True)

        df = df[['累积净值', '收益回撤比', '最大回撤']]
        # df.rename(columns={'累积净值': f'{sheet_df.loc[conf.get_fullname(), factor]}累积净值',
        #                    '收益回撤比': f'{sheet_df.loc[conf.get_fullname(), factor]}收益回撤比',
        #                    '最大回撤': f'{sheet_df.loc[conf.get_fullname(), factor]}最大回撤',
        #                    }, inplace=True)
        # df.rename(columns={'累积净值': sheet_df.loc[conf.get_fullname(), f'#FACTOR-{factor}累积净值'],
        #                    '收益回撤比': sheet_df.loc[conf.get_fullname(), f'#FACTOR-{factor}收益回撤比'],
        #                    '最大回撤': sheet_df.loc[conf.get_fullname(), f'#FACTOR-{factor}最大回撤'],
        #                    }, inplace=True)
        df.rename(columns={'累积净值': sheet_df.loc[conf.get_fullname(), f'#FACTOR-{factor}'] + '累积净值',
                           '收益回撤比': sheet_df.loc[conf.get_fullname(), f'#FACTOR-{factor}'] + '收益回撤比',
                           '最大回撤': sheet_df.loc[conf.get_fullname(), f'#FACTOR-{factor}'] + '最大回撤',
                           }, inplace=True)

        if all_df.empty:
            all_df = df.copy()
        else:
            all_df = all_df.merge(df, left_index=True, right_index=True, how='left')

    all_df.index.name = None
    all_df = all_df.T
    # for col in all_df.columns:
    #     all_df[col] = all_df[col].str.replace('%', '').astype(float) / 100
    filter = ''
    if strategy_list[0]['is_use_spot']:
        pre = 'Spot'
    else:
        pre = 'Swap'

    if 'filter_list' in strategy_list[0]:
        if len(strategy_list[0]['filter_list']) > 0:
            filter += '_F'
            for f in strategy_list[0]['filter_list']:
                filter += f"{f}"

    if 'long_filter_list' in strategy_list[0] or 'short_filter_list' in strategy_list[0]:
        if len(strategy_list[0]['long_filter_list']) > 0:
            filter += '_LF'
            for f in strategy_list[0]['long_filter_list']:
                filter += f"{f}"
        if len(strategy_list[0]['short_filter_list']) > 0:
            filter += '_SF'
            for f in strategy_list[0]['short_filter_list']:
                filter += f"{f}"

    res_name = f"{pre}_{factor}_L{strategy_list[0]['long_select_coin_num']}S{strategy_list[0]['short_select_coin_num']}_{strategy_list[0]['factor_list'][0][1]}{filter}"

    all_df.to_csv(get_file_path('data', '遍历结果', f'{res_name}.csv'))
    print(all_df)

    logger.ok(f'完成参数平原结果输出，花费时间：{time.time() - s_time:.3f}秒')

    # 绘图
    some_folder = list(backtest_factory.result_folders)[0]  # 拿出factory中任意一个config使用的folder
    plotly_plot(all_df, some_folder, f'参数平原图_{res_name}')


if __name__ == '__main__':
    divider(f'版本: {sys_version}，当前时间:', '#', _logger=logger)
    logger.debug(f'BUILD VERSION: {build_version}')
    logger.info(f'系统启动中，稍等...')

    # ====================================================================================================
    # 1. 配置需要遍历的参数
    # ====================================================================================================
    backtest_factor = list(factor_param_range_dict.keys())[0]

    # 因子遍历的参数范围
    # factor_param_range_dict = {
    #     'QuoteVolumeMean': [_ for _ in range(3, 31, 1)],
    # }

    # # 查看参数平原的图表类型
    # # - Y  每年参数平原
    # # - M  每月参数平原
    # # - Q  每季参数平原
    # params_plot_type = 'Y'

    # ====================================================================================================
    # 2. 执行遍历
    # ====================================================================================================
    run(backtest_factor, params_plot_type)
