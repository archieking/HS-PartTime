"""
寻找最优参数
"""
import warnings

import pandas as pd

from core.backtest import find_best_params
from core.model.backtest_config import BacktestConfigFactory
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
# ** 重要的函数区域 **
# 负责我们保温杯系统平顺执行的核心组件
# ====================================================================================================
# region 重要的函数区域
# endregion


if __name__ == '__main__':
    divider(f'版本: {sys_version}，当前时间:', '#', _logger=logger)
    logger.debug(f'BUILD VERSION: {build_version}')
    logger.info(f'系统启动中，稍等...')

    # ====================================================================================================
    # ** 参数遍历配置 **
    # 会根据一下遍历参数的配置，动态生成回测组
    # ====================================================================================================
    # 因子遍历的参数范围
    factor_param_range_dict = {
        'ILLQStd': [_ for _ in range(6, 10, 1)],
        'PctChange': [_ for _ in range(6, 10, 1)],
    }
    # ################################################################################################
    # 情况1: 如果说你定义了你策略用不上的配置，则不会自动遍历
    # 比如：策略配置为 Strategy_QuoteVolumeMean，
    # 但是factor_param_range_dict中没有定义 QuoteVolumeMean，那么不会自动遍历

    # 情况2: 周期不能混合
    # hold_period = 日线不能和小时级别混合
    # ###############################################################################################
    backtest_factory = BacktestConfigFactory.init(factor_param_range_dict=factor_param_range_dict)

    print('生成回测配置...')
    backtest_factory.generate_configs()

    # 是否展示热力图
    # - 仅仅只在双因子的模式下生效，不支持多因子，超过不绘图
    # - 双因子模式下，热力图仅仅展示累积净值
    # - 双因子模式，指的是选币单因子 & 过滤单因子
    is_show_heat_map = True

    # 开始寻找最优参数
    find_best_params(backtest_factory, is_show_heat_map)
