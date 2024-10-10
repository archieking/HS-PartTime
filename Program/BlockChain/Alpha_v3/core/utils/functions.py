"""
选币策略框架
"""
import hashlib
import shutil

from core.model.backtest_config import BacktestConfig
from core.utils.path_kit import get_file_path

# -*- coding: utf-8 -*-
"""
选币策略框架
"""
import gc
import warnings
from pathlib import Path
from typing import Dict

import numba as nb
import numpy as np
import pandas as pd

from config import stable_symbol, backtest_path, swap_path, spot_path

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
    min_qty_df = pd.read_csv(file_path, encoding='gbk')
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
    print('清理数据缓存')
    cache_path = get_file_path('data', 'cache', as_path_type=True)
    if cache_path.exists():
        shutil.rmtree(cache_path)

    print('加载现货和合约数据...')
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


@nb.njit
def super_fast_groupby_and_rank(values, if_reverse):
    """
    实现逻辑：
    1. 对每个组进行排序。
    2. 找到每个组内相同值的元素。
    3. 为这些相同值的元素分配相同的最小排名。

    :param values:
    :param if_reverse:
    :return:
    """
    # 对 'candle_begin_time' 进行排序，确保分组时是有序的
    sorted_idx = np.lexsort((values[:, 0], values[:, 1]))
    sorted_values = values[sorted_idx]

    # 分组操作
    unique_times, group_indices = np.unique(sorted_values[:, 1], return_index=True)

    # 初始化排名数组
    ranks = np.empty_like(sorted_values[:, 0], dtype=np.float64)

    # 逐组处理并计算排名
    for i in range(len(group_indices)):
        start_idx = group_indices[i]
        end_idx = group_indices[i + 1] if i + 1 < len(group_indices) else len(sorted_values)
        group_values = sorted_values[start_idx:end_idx, 0]

        # 使用 numpy 的 argsort 来排序，并计算排名
        sorted_order = np.argsort(group_values)
        if if_reverse:
            sorted_order = sorted_order[::-1]

        ranks[start_idx:end_idx] = np.argsort(sorted_order) + 1  # 计算排名，+1 是为了从 1 开始

    # 还原排名结果到原始顺序
    final_ranks = np.empty_like(ranks)
    final_ranks[sorted_idx] = ranks

    return final_ranks


def check_md5(hash_file: Path, factor_df_md5: str) -> bool:
    # 检查是否存在hash校验文件
    if hash_file.exists():
        hash_val = hash_file.read_text(encoding='utf8')
        return factor_df_md5 == hash_val

    return False


def save_md5(hash_file: Path, factor_df_md5: str) -> None:
    hash_file.write_text(factor_df_md5, encoding='utf8')


def calc_factor_md5(df: pd.DataFrame, data_size: int = 100) -> str:
    hash_object = hashlib.md5()
    # 将每个块转换为CSV格式的字符串，不包含索引和列名
    chunk_str = df.tail(data_size).to_csv(index=False, header=False).encode('utf-8')
    # 更新哈希对象的数据
    hash_object.update(chunk_str)
    factor_df_md5 = hash_object.hexdigest()

    return factor_df_md5


def save_performance_df_csv(conf: BacktestConfig, **kwargs):
    for name, df in kwargs.items():
        file_path = conf.get_result_folder() / f'{name}.csv'
        df.to_csv(file_path, encoding='gbk')
