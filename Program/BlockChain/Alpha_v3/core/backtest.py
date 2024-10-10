import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from config import job_num, raw_data_path, backtest_path, backtest_iter_path
from core.equity import calc_equity, show_plot_performance
from core.figure import mat_heatmap
from core.model.backtest_config import BacktestConfig
from core.model.backtest_config import BacktestConfigFactory
from core.model.timing_signal import TimingSignal
from core.select_coin import calc_factors, select_coins, concat_select_results, process_select_results, \
    agg_multi_strategy_ratio
from core.utils.functions import load_spot_and_swap_data, save_performance_df_csv
from core.utils.log_kit import logger, divider


def step2_load_data(conf: BacktestConfig):
    """
    读取回测所需数据，并做简单的预处理
    :param conf:
    :return:
    """
    logger.info(f'读取数据中心数据...')
    s_time = time.time()

    # 读取数据
    # 针对现货策略和非现货策略读取的逻辑完全不同。
    # - 如果是纯合约模式，只需要读入 swap 数据并且合并即可
    # - 如果是现货模式，需要读入 spot 和 swap 数据并且合并，然后添加 tag
    load_spot_and_swap_data(conf)  # 串行方式，完全等价
    logger.ok(f'完成读取数据中心数据，花费时间：{time.time() - s_time:.2f}秒')


def step3_calc_factors(conf: BacktestConfig):
    """
    计算因子
    :param conf: 配置
    :return:
    """
    s_time = time.time()
    logger.info(f'因子计算...')
    calc_factors(conf)
    logger.ok(f'完成计算因子，花费时间：{time.time() - s_time:.2f}秒')


def step4_select_coins(conf: BacktestConfig):
    """
    选币
    :param conf: 配置
    :return:
    """
    s_time = time.time()
    logger.info(f'选币...')
    select_coins(conf)  # 选币
    logger.ok(f'完成选币，花费时间：{time.time() - s_time:.3f}秒')


def step5_aggregate_select_results(conf: BacktestConfig, save_final_result=False):
    logger.info(f'整理选币结果...')
    # 整理选币结果
    concat_select_results(conf)  # 合并多个策略的选币结果
    select_results = process_select_results(conf)  # 生成整理后的选币结果
    logger.debug(
        f'选币结果df大小：{select_results.memory_usage(deep=True).sum() / 1024 / 1024:.4f} M\n'
        f'选币结果：\n{select_results}', )
    if save_final_result:
        # 存储最终的选币结果
        select_results.to_pickle(backtest_path / 'final_select_results.pkl')

    # 聚合大杂烩中多策略的权重，以及多offset选币的权重聚合
    s_time = time.time()
    logger.debug('开始权重聚合...')
    df_spot_ratio, df_swap_ratio = agg_multi_strategy_ratio(conf, select_results)
    logger.ok(f'完成权重聚合，花费时间： {time.time() - s_time:.3f}秒')

    return df_spot_ratio, df_swap_ratio


def step6_simulate_performance(conf: BacktestConfig, df_spot_ratio, df_swap_ratio, pivot_dict_spot, pivot_dict_swap,
                               if_show_plot=False):
    logger.info(f'开始模拟交易，累计回溯 {len(df_spot_ratio):,} 小时（~{len(df_spot_ratio) / 24:,.0f}天）...')
    account_df, rtn, year_return, month_return, quarter_return = calc_equity(
        conf, pivot_dict_spot, pivot_dict_swap,
        df_spot_ratio, df_swap_ratio
    )
    save_performance_df_csv(conf,
                            资金曲线=account_df,
                            策略评价=rtn,
                            年度账户收益=year_return,
                            季度账户收益=quarter_return,
                            月度账户收益=month_return)

    has_timing_signal = isinstance(conf.timing, TimingSignal)

    if has_timing_signal:
        account_df, rtn, year_return = simu_re_timing(conf, df_spot_ratio, df_swap_ratio, pivot_dict_spot,
                                                      pivot_dict_swap)

    if if_show_plot:
        show_plot_performance(conf, account_df, rtn, year_return, '再择时: ')

    return conf.report


def simu_re_timing(conf: BacktestConfig, df_spot_ratio, df_swap_ratio, pivot_dict_spot, pivot_dict_swap):
    s_time = time.time()
    logger.info(f'{conf.get_fullname(as_folder_name=True)} 资金曲线择时，生成动态杠杆')

    account_df = pd.read_csv(conf.get_result_folder() / '资金曲线.csv', index_col=0, encoding='gbk')

    leverage = conf.timing.get_dynamic_leverage(account_df['equity'])
    logger.ok(f'完成生成动态杠杆，花费时间： {time.time() - s_time:.3f}秒')

    s_time = time.time()
    logger.info(
        f'开始动态杠杆再择时模拟交易，累计回溯 {len(df_spot_ratio):,} 小时（~{len(df_spot_ratio) / 24:,.0f}天）...')

    account_df, rtn, year_return, month_return, quarter_return = calc_equity(conf, pivot_dict_spot, pivot_dict_swap,
                                                                             df_spot_ratio, df_swap_ratio, leverage)
    save_performance_df_csv(conf,
                            资金曲线_再择时=account_df,
                            策略评价_再择时=rtn,
                            年度账户收益_再择时=year_return,
                            季度账户收益_再择时=quarter_return,
                            月度账户收益_再择时=month_return)

    logger.ok(f'完成动态杠杆再择时模拟交易，花费时间：{time.time() - s_time:.3f}秒')

    return account_df, rtn, year_return


def simu_performance_on_select(conf: BacktestConfig):
    import logging
    logger.setLevel(logging.WARNING)  # 可以减少中间输出的log
    logger.debug(conf.get_fullname())
    # ====================================================================================================
    # 5. 整理大杂烩选币结果
    # - 把大杂烩中每一个策略的选币结果聚合成一个df
    # ====================================================================================================
    df_spot_ratio, df_swap_ratio = step5_aggregate_select_results(conf)

    pivot_dict_spot = pd.read_pickle(raw_data_path / 'market_pivot_spot.pkl')
    pivot_dict_swap = pd.read_pickle(raw_data_path / 'market_pivot_swap.pkl')
    return step6_simulate_performance(conf, df_spot_ratio, df_swap_ratio, pivot_dict_spot, pivot_dict_swap)


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

    # 删除缓存
    # shutil.rmtree(backtest_path, ignore_errors=True)
    backtest_path.mkdir(parents=True, exist_ok=True)

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
    pivot_dict_spot = pd.read_pickle(raw_data_path / 'market_pivot_spot.pkl')
    pivot_dict_swap = pd.read_pickle(raw_data_path / 'market_pivot_swap.pkl')
    step6_simulate_performance(conf, df_spot_ratio, df_swap_ratio, pivot_dict_spot, pivot_dict_swap, if_show_plot=True)
    logger.ok(f'完成，回测时间：{time.time() - r_time:.3f}秒')


# ====================================================================================================
# ** 回测主程序 **
# 1. 准备工作
# 2. 读取数据
# 3. 计算因子
# 4. 并行选币并且计算资金曲线
# 5. 根据回测参数列表，展示最优参数
# ====================================================================================================
def find_best_params(factory: BacktestConfigFactory, show_heat_map=False):
    # ====================================================================================================
    # 1. 准备工作
    # ====================================================================================================
    divider('参数遍历开始', '*')
    iter_results_folder = backtest_iter_path / factory.backtest_name

    # 删除缓存
    shutil.rmtree(iter_results_folder, ignore_errors=True)

    conf_list = factory.config_list
    for index, conf in enumerate(conf_list):
        logger.debug(f'参数组合{index + 1}｜共{len(conf_list)}')
        logger.debug(f'{conf.get_fullname()}')
        conf.save()
        print()
    logger.ok('一共需要回测的参数组合数：{}'.format(len(conf_list)))

    # 记录一下时间戳
    r_time = time.time()

    # ====================================================================================================
    # 2. 读取回测所需数据，并做简单的预处理
    # ====================================================================================================
    logger.info(f'读取数据...')
    s_time = time.time()
    conf = factory.generate_all_factor_config()

    # 读取数据
    # 针对现货策略和非现货策略读取的逻辑完全不同。
    # - 如果是纯合约模式，只需要读入 swap 数据并且合并即可
    # - 如果是现货模式，需要读入 spot 和 swap 数据并且合并，然后添加 tag
    load_spot_and_swap_data(conf)  # 串行方式，完全等价
    logger.ok(f'完成读取数据中心数据，花费时间：{time.time() - s_time:.3f}秒')

    # ====================================================================================================
    # 3. 计算因子
    # ====================================================================================================
    s_time = time.time()
    logger.info(f'因子计算...')
    calc_factors(conf)
    logger.ok(f'完成计算因子，花费时间：{time.time() - s_time:.3f}秒，累计时间：{(time.time() - r_time):.3f}秒')

    # ====================================================================================================
    # 4. 选币
    # - 注意：选完之后，每一个策略的选币结果会被保存到硬盘
    # ====================================================================================================
    s_time = time.time()
    logger.info(f'选币（时间最久，耐心等等）...')
    select_coins(factory.config_list)  # 选币
    logger.ok(f'完成选币，花费时间：{time.time() - s_time:.3f}秒，累计时间：{(time.time() - r_time):.3f}秒')

    # ====================================================================================================
    # 5. 针对选币结果进行聚合、回测模拟
    # ====================================================================================================
    logger.info(f'回测模拟（时间会比较久）...')
    logger.debug(f'并行任务数：{job_num}')
    s_time = time.time()
    report_list = []

    # 串行
    # for conf in conf_list:
    #     report_list.append(simulate_performance(conf))

    # 并行
    with ProcessPoolExecutor(max_workers=job_num) as executor:
        futures = [executor.submit(simu_performance_on_select, conf) for conf in conf_list]
        for future in tqdm(as_completed(futures), total=len(conf_list), desc='回测进度'):
            try:
                report = future.result()
                report_list.append(report)
                if len(report_list) > 65535:
                    logger.debug(f'回测报表数量为 {len(report_list)}，超过 65535，后续可能会占用海量内存')
            except Exception as e:
                logger.exception(e)
                exit(1)
    logger.ok(f'回测模拟币完成，花费时间：{time.time() - s_time:.3f}秒，累计时间：{(time.time() - r_time):.3f}秒')

    # ====================================================================================================
    # 6. 根据回测参数列表，展示最优参数
    # ====================================================================================================
    s_time = time.time()
    logger.info(f'展示最优参数...')
    if len(report_list) > 65535:
        logger.warning(f'回测参数列表超过 65535，会占用海量内存，请手动合并 `data -> 遍历结果` 下完整的回测结果')
        return None

    all_params_map = pd.concat(report_list, ignore_index=True)
    report_columns = all_params_map.columns  # 缓存列名

    # 合并参数细节
    sheet = factory.get_name_params_sheet()
    if len(sheet.columns) > 1023:
        logger.warning(f'回测参数列表超过 1023，会占用海量内存，合并结果中不再包含因子列')
    else:
        all_params_map = all_params_map.merge(sheet, left_on='param', right_on='fullname', how='left')

    # 按照累积净值排序，并整理结果
    all_params_map.sort_values(by='累积净值', ascending=False, inplace=True)
    all_params_map = all_params_map[[*sheet.columns, *report_columns]].drop(columns=['param'])
    all_params_map.to_excel(iter_results_folder / f'最优参数.xlsx', index=False)
    print(all_params_map)
    logger.ok(f'完成展示最优参数，花费时间：{time.time() - s_time:.3f}秒，累计时间：{(time.time() - r_time):.3f}秒')

    # ====================================================================================================
    # 7. 根据回测参数列表，展示热力图。仅仅只支持双因子
    # ====================================================================================================
    s_time = time.time()
    if not show_heat_map or len(list(filter(lambda x: x.startswith('#FACTOR-'), sheet.columns))) > 2:
        return all_params_map

    # 热力图参数
    indicator = '累积净值'

    series_col = []
    for col in list(sheet.columns) + [indicator]:
        if not col.startswith('#FACTOR-'):
            continue
        all_params_map[col] = all_params_map[col].astype(float)
        series_col.append(col)

    pivot_table = pd.pivot_table(all_params_map,
                                 values=indicator,
                                 index=all_params_map[series_col[0]],
                                 columns=all_params_map[series_col[1]])
    pivot_table = pivot_table.astype(float)  # 红军同学指出，保障作图的功能性
    mat_heatmap(pivot_table, f'热力图 {indicator}')

    logger.info(f'完成展示双因子热力图，花费时间：{time.time() - s_time:.3f}秒，累计时间：{(time.time() - r_time):.3f}秒')

    return all_params_map
