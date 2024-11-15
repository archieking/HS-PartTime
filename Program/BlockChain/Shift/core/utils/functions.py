"""
仓位管理框架
"""

import gc
import shutil
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from config import stable_symbol, swap_path, spot_path
from core.model.backtest_config import BacktestConfig
from core.utils.log_kit import logger
from core.utils.path_kit import get_file_path

warnings.filterwarnings('ignore')


# =====策略相关函数
def del_insufficient_data(symbol_candle_data) -> Dict[str, pd.DataFrame]:
    """
    删除数据长度不足的币种信息

    :param symbol_candle_data:
    :return
    """
    # ===删除成交量为0的线数据、k线数不足的币种
    symbol_list = list(symbol_candle_data.keys())
    for symbol in symbol_list:
        # 删除空的数据
        if symbol_candle_data[symbol] is None or symbol_candle_data[symbol].empty:
            del symbol_candle_data[symbol]
            continue
        # 删除该币种成交量=0的k线
        # symbol_candle_data[symbol] = symbol_candle_data[symbol][symbol_candle_data[symbol]['volume'] > 0]

    return symbol_candle_data


def ignore_error(anything):
    return anything


def load_min_qty(file_path: Path) -> (int, Dict[str, int]):
    # 读取min_qty文件并转为dict格式
    min_qty_df = pd.read_csv(file_path, encoding='utf-8-sig')
    min_qty_df['最小下单量'] = -np.log10(min_qty_df['最小下单量']).round().astype(int)
    default_min_qty = min_qty_df['最小下单量'].max()
    min_qty_df.set_index('币种', inplace=True)
    min_qty_dict = min_qty_df['最小下单量'].to_dict()

    return default_min_qty, min_qty_dict


def is_trade_symbol(symbol, black_list, white_list) -> bool:
    """
    过滤掉不能用于交易的币种，比如稳定币、非USDT交易对，以及一些杠杆币
    :param symbol: 交易对
    :param black_list: 黑名单
    :param white_list: 白名单
    :return: 是否可以进入交易，True可以参与选币，False不参与
    """
    if white_list:
        if symbol in white_list:
            return True
        else:
            return False

    # 稳定币和黑名单币不参与
    if not symbol or not symbol.endswith('USDT') or symbol in black_list:
        return False

    # 筛选杠杆币
    base_symbol = symbol.upper().replace('-USDT', 'USDT')[:-4]
    if base_symbol.endswith(('UP', 'DOWN', 'BEAR', 'BULL')) and base_symbol != 'JUP' or base_symbol in stable_symbol:
        return False
    else:
        return True


def load_spot_and_swap_data(conf: BacktestConfig) -> (pd.DataFrame, pd.DataFrame):
    """
    加载现货和合约数据
    :param conf: 回测配置
    :return:
    """
    logger.debug('清理数据缓存')
    cache_path = get_file_path('data', 'cache', as_path_type=True)
    if cache_path.exists():
        shutil.rmtree(cache_path)

    logger.debug('加载现货和合约数据...')
    # 读入合约数据
    symbol_swap_candle_data = pd.read_pickle(swap_path)
    # 过滤掉不能用于交易的币种
    symbol_swap_candle_data = {k: v for k, v in symbol_swap_candle_data.items()
                               if is_trade_symbol(k, conf.black_list, conf.white_list)}

    # 过滤掉数据不足的币种
    all_candle_df_list = list(del_insufficient_data(symbol_swap_candle_data).values())
    all_symbol_list = set(symbol_swap_candle_data.keys())

    # 读入现货数据
    if conf.is_use_spot:
        symbol_spot_candle_data = pd.read_pickle(spot_path)
        # 过滤掉不能用于交易的币种
        symbol_spot_candle_data = {k: v for k, v in symbol_spot_candle_data.items()
                                   if is_trade_symbol(k, conf.black_list, conf.white_list)}

        # 过滤掉数据不足的币种
        all_candle_df_list = all_candle_df_list + list(del_insufficient_data(symbol_spot_candle_data).values())
        all_symbol_list = list(all_symbol_list | set(symbol_spot_candle_data.keys()))
        del symbol_spot_candle_data

    # 保存数据
    pkl_path = get_file_path('data', 'cache', 'all_candle_df_list.pkl')
    pd.to_pickle(all_candle_df_list, pkl_path)

    del symbol_swap_candle_data
    del all_candle_df_list

    gc.collect()

    return tuple(all_symbol_list)  # 节省内存，包装成tuple


def save_performance_df_csv(conf: BacktestConfig, **kwargs):
    for name, df in kwargs.items():
        file_path = conf.get_result_folder() / f'{name}.csv'
        df.to_csv(file_path, encoding='utf-8-sig')
