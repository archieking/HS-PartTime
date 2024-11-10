"""
选币策略框架
"""
import shutil
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

from config import backtest_path, backtest_iter_path, backtest_name
from core.model.account_type import AccountType
from core.model.rebalance_mode import RebalanceMode
from core.model.strategy_config import StrategyConfig, FactorConfig, FilterFactorConfig
from core.model.timing_signal import TimingSignal
from core.utils.log_kit import logger as default_logger
from core.utils.path_kit import get_folder_path
from core.utils.strategy_hub import StrategyHub


class BacktestConfig:
    data_file_fingerprint: str = ''  # 记录数据文件的指纹

    def __init__(self, name: str, **conf):
        self.name: str = name  # 账户名称，建议用英文，不要带有特殊符号
        self.logger = conf.get("logger", default_logger)

        self.start_date: str = conf.get("start_date", '2021-01-01')  # 回测开始时间
        self.end_date: str = conf.get("end_date", '2024-03-30')  # 回测结束时间

        # 账户回测交易模拟配置
        self.account_type: AccountType = AccountType.translate(conf.get("account_type", '普通账户'))  # 账户类型
        self.rebalance_mode: RebalanceMode = RebalanceMode.init(conf.get('rebalance_mode', None))
        self.avg_price_col: str = conf.get("avg_price_col", 'avg_price_1m')  # 平均成交价格
        self.initial_usdt: int | float = conf.get("initial_usdt", 10000)  # 初始现金
        self.leverage: int | float = conf.get("leverage", 1)  # 杠杆数。我看哪个赌狗要把这里改成大于1的。高杠杆如梦幻泡影。不要想着一夜暴富，脚踏实地赚自己该赚的钱。
        self.margin_rate = conf.get('margin_rate', 0.05)  # 维持保证金率，净值低于这个比例会爆仓

        self.swap_c_rate: float = conf.get("swap_c_rate", 6e-4)  # 合约买卖手续费
        self.spot_c_rate: float = conf.get("spot_c_rate", 2e-3)  # 现货买卖手续费

        self.swap_min_order_limit: int | float = conf.get("swap_min_order_limit", 5)  # 合约最小下单量
        self.spot_min_order_limit: int | float = conf.get("spot_min_order_limit", 10)  # 现货最小下单量

        # 策略配置
        self.black_list: List[str] = conf.get('black_list',
                                              [])  # 拉黑名单，永远不会交易。不喜欢的币、异常的币。例：LUNA-USDT, 这里与实盘不太一样，需要有'-'
        self.white_list: List[str] = conf.get('white_list',
                                              [])  # 如果不为空，即只交易这些币，只在这些币当中进行选币。例：LUNA-USDT, 这里与实盘不太一样，需要有'-'
        self.min_kline_num: int = conf.get('min_kline_num', 168)  # 最少上市多久，不满该K线根数的币剔除，即剔除刚刚上市的新币。168：标识168个小时，即：7*24

        # 再择时配置
        self.timing: Optional[TimingSignal] = None

        self.is_use_spot: bool = False  # 是否包含现货策略
        self.is_day_period: bool = False  # 是否是日盘，否则是小时盘
        self.is_hour_period: bool = False  # 是否是小时盘，否则是日盘
        self.factor_params_dict: Dict[str, set] = {}
        self.factor_col_name_list: List[str] = []
        self.max_hold_period: str = '1H'  # 最大的持仓周期，默认值设置为最小
        self.hold_period_list: List[str] = []  # 持仓周期列表
        self.max_offset_len: int = 0

        # 策略列表，包含每个策略的详细配置
        self.strategy_list: List[StrategyConfig] = []
        self.strategy_name_list: List[str] = []
        self.strategy_list_raw: List[dict] = []

        # 策略评价
        self.report: Optional[pd.DataFrame] = None

        # 遍历标记
        self.iter_round: int | str = 0  # 遍历的INDEX，0表示非遍历场景，从1、2、3、4、...开始表示是第几个循环，当然也可以赋值为具体名称

    def __repr__(self):
        return f"""{'+' * 56}
# {self.name} 配置信息如下：
+ 回测时间: {self.start_date} ~ {self.end_date}
+ 手续费: 合约{self.swap_c_rate * 100:.2f}%，现货{self.spot_c_rate * 100:.2f}%
+ 杠杆: {self.leverage:.2f}
+ 最小K线数量: {self.min_kline_num}
+ 维持保证金率: {self.margin_rate * 100:.2f}%
+ 拉黑名单: {self.black_list}，只交易名单: {self.white_list}
+ Rebalance 模式: {self.rebalance_mode}
+ 再择时: {self.timing}
{''.join([str(item) for item in self.strategy_list])}
{'+' * 56}
"""

    @property
    def hold_period_type(self):
        return 'D' if self.is_day_period else 'H'

    def info(self):
        # 输出一下配置信息
        self.logger.debug(self)

    def get_fullname(self, as_folder_name=False):
        fullname_list = [self.name]
        for stg in self.strategy_list:
            fullname_list.append(f"{stg.get_fullname(as_folder_name)}")

        if self.timing:
            fullname_list.append(f'再择时:{self.timing}')

        fullname = ' '.join(fullname_list)
        return f'{self.name}' if as_folder_name else fullname

    def load_strategy_config(self, strategy_list: list | tuple, re_timing_config=None):
        self.strategy_list_raw = strategy_list
        # 所有策略中的权重
        all_cap_weight = sum(item["cap_weight"] for item in strategy_list)

        for index, stg_dict in enumerate(strategy_list):
            # 更新策略权重
            strategy_name = stg_dict['strategy']

            stg_cfg = StrategyConfig.init(index, file=StrategyHub.get_by_name(strategy_name), **stg_dict)

            offset_list = list(filter(lambda x: x < stg_cfg.period_num, stg_cfg.offset_list))
            if len(offset_list) != len(stg_cfg.offset_list):
                self.logger.warning(
                    f'策略{stg_cfg.name}的offset_list设置有问题，自动裁剪。原始值：{stg_cfg.offset_list},裁剪后：{offset_list}')
            stg_cfg.offset_list = offset_list
            stg_cfg.cap_weight = stg_cfg.cap_weight / all_cap_weight

            if stg_cfg.is_day_period:
                self.is_day_period = True
            else:
                self.is_hour_period = True

            # 缓存持仓周期的事情
            if stg_cfg.hold_period not in self.hold_period_list:
                self.hold_period_list.append(stg_cfg.hold_period)
                # 更新最大的持仓周期
                if pd.to_timedelta(self.max_hold_period) < pd.to_timedelta(stg_cfg.hold_period):
                    self.max_hold_period = stg_cfg.hold_period

            self.is_use_spot = self.is_use_spot or stg_cfg.is_use_spot
            if self.is_use_spot and self.leverage >= 2:
                self.logger.error(f'现货策略不支持杠杆大于等于2的情况，请重新配置')
                exit(1)

            if stg_cfg.long_select_coin_num == 0 and (stg_cfg.short_select_coin_num == 0 or
                                                      stg_cfg.short_select_coin_num == 'long_nums'):
                self.logger.warning('策略中的选股数量都为0，忽略此策略配置')
                continue

            self.strategy_list.append(stg_cfg)
            self.strategy_name_list.append(stg_cfg.name)
            self.factor_col_name_list += stg_cfg.factor_columns

            # 针对当前策略的因子信息，整理之后的列名信息，并且缓存到全局
            for factor_config in stg_cfg.all_factors:
                # 添加到并行计算的缓存中
                if factor_config.name not in self.factor_params_dict:
                    self.factor_params_dict[factor_config.name] = set()
                self.factor_params_dict[factor_config.name].add(factor_config.param)

            if len(stg_cfg.offset_list) > self.max_offset_len:
                self.max_offset_len = len(stg_cfg.offset_list)

        self.factor_col_name_list = list(set(self.factor_col_name_list))

        if all((self.is_hour_period, self.is_day_period)):
            self.logger.critical(f'策略中同时存在小时线和日线的策略融合，请检查配置')
            exit()

        if re_timing_config:
            self.timing = TimingSignal(**re_timing_config)

    @classmethod
    def init_from_config(cls, load_strategy_list: bool = True) -> "BacktestConfig":
        import config

        backtest_config = cls(
            config.backtest_name,
            account_type=config.account_type,  # 账户类型
            rebalance_mode=getattr(config, 'rebalance_mode', None),  # rebalance类型
            start_date=config.start_date,  # 回测开始时间
            end_date=config.end_date,  # 回测结束时间
            # ** 交易配置 **
            initial_usdt=config.initial_usdt,  # 初始usdt
            leverage=config.leverage,  # 杠杆
            margin_rate=config.margin_rate,  # 维持保证金率
            swap_c_rate=config.swap_c_rate,  # 合约买入手续费
            spot_c_rate=config.spot_c_rate,  # 现货买卖手续费
            spot_min_order_limit=config.spot_min_order_limit,  # 现货最小下单量
            swap_min_order_limit=config.swap_min_order_limit,  # 合约最小下单量
            # ** 数据参数 **
            avg_price_col=config.avg_price_col,  # 平均价格列名
            black_list=config.black_list,  # 拉黑名单
            white_list=config.white_list,  # 只交易名单
            min_kline_num=config.min_kline_num,  # 最小K线数量，k线数量少于这个数字的部分不会计入计算
        )

        # ** 策略配置 **
        # 初始化策略，默认都是需要初始化的
        if load_strategy_list:
            re_timing_config = getattr(config, 're_timing', None)  # 从config中读取选币再择时的策略配置
            backtest_config.load_strategy_config(config.strategy_list, re_timing_config)

        return backtest_config

    def set_report(self, report: pd.DataFrame):
        report['param'] = self.get_fullname()
        self.report = report

    def get_result_folder(self) -> Path:
        if self.iter_round == 0:
            return get_folder_path(backtest_path, self.get_fullname(as_folder_name=True), as_path_type=True)
        else:
            return get_folder_path(
                backtest_iter_path,
                self.name,
                f'参数组合_{self.iter_round}' if isinstance(self.iter_round, int) else self.iter_round,
                as_path_type=True
            )

    def get_strategy_config_sheet(self, with_factors=True, sep_filter=False) -> dict:
        factor_dict = {}
        for stg in self.strategy_list:
            for attr_in in ['hold_period', 'is_use_spot', 'offset_list', 'cap_weight']:
                if attr_in not in factor_dict:
                    factor_dict[attr_in] = []
                factor_dict[attr_in].append(getattr(stg, attr_in))

            for factor_config in stg.all_factors:
                if sep_filter:
                    factor_type = 'FACTOR' if isinstance(factor_config, FactorConfig) else 'FILTER'
                    _name = f'#{factor_type}-{factor_config.name}'
                else:
                    _name = f'#FACTOR-{factor_config.name}'
                _val = factor_config.param
                if _name not in factor_dict:
                    factor_dict[_name] = []
                factor_dict[_name].append(_val)
        ret = {
            '策略': self.name,
            'fullname': self.get_fullname(),
        }
        if with_factors:
            ret.update(**{
                k: "_".join(map(str, v)) for k, v in factor_dict.items()
            })

        if self.timing:
            ret['再择时'] = str(self.timing)
        return ret

    def save(self):
        pd.to_pickle(self, self.get_result_folder() / 'config.pkl')

    def delete_cache(self):
        shutil.rmtree(self.get_result_folder())


class BacktestConfigFactory:
    """
    遍历参数的时候，动态生成配置
    """
    STRATEGY_FACTOR_ATTR = [
        'factor_list',
        'long_factor_list',
        'short_factor_list',
        'filter_list',
        'long_filter_list',
        'short_filter_list',
        'filter_list_post',
        'long_filter_list_post',
        'short_filter_list_post',
    ]

    def __init__(self, **conf):
        # ====================================================================================================
        # ** 参数遍历配置 **
        # 可以指定因子遍历的参数范围
        # ====================================================================================================
        self.factor_param_range_dict: dict = conf.get("factor_param_range_dict", {})
        self.strategy_param_range_dict: dict = conf.get("strategy_param_range_dict", {})
        self.default_param_range = conf.get("default_param_range", [])
        self.backtest_name = conf.get("backtest_name", backtest_name)

        if not self.backtest_name:
            self.backtest_name = f'默认策略-{datetime.now().strftime("%Y%m%dT%H%M%S")}'

        # 缓存全局配置
        self.is_use_spot = conf.get("is_use_spot", False)
        self.black_list = conf.get("black_list", set())

        # 存储生成好的config list和strategy list
        self.config_list: List[BacktestConfig] = []
        self.strategy_list: List[StrategyConfig] = []

        default_logger.info(f'遍历结果输出路径：{self.result_folder}')

    @classmethod
    def init(cls, **conf):
        return cls(
            factor_param_range_dict=conf.get("factor_param_range_dict", {}),
            strategy_param_range_dict=conf.get("strategy_param_range_dict", {}),
            default_param_range=conf.get("default_param_range", []),
            backtest_name=conf.get("backtest_name", backtest_name),
        )

    @property
    def result_folder(self) -> Path:
        return get_folder_path(backtest_iter_path, self.backtest_name, as_path_type=True)

    def update_meta_by_config(self, config: BacktestConfig):
        """
        # 缓存是否使用现货等状态
        :param config: 生成的配置信息
        :return: None
        """
        self.is_use_spot = self.is_use_spot or config.is_use_spot
        self.black_list = self.black_list | set(config.black_list)

    def get_candidates_by_factors(self, strategy_dict, param_name, target_factors, val_index) -> List[tuple]:
        """
        根据指定的因子，获取所有的排列组合
        :param strategy_dict:
        :param param_name:
        :param target_factors:
        :param val_index:
        :return:
        """
        if param_name in ('factor_list', 'filter_list', 'filter_list_post'):
            if f'long_{param_name}' in strategy_dict or f'short_{param_name}' in strategy_dict:
                # 如果设置过的话，默认单边是挂空挡
                legacy_strategy = None
            else:
                legacy_strategy = StrategyHub.get_by_name(strategy_dict['strategy'])
        else:
            legacy_strategy = StrategyHub.get_by_name(strategy_dict['strategy'])

        factor_tuple_list = strategy_dict.get(param_name,
                                              getattr(legacy_strategy, param_name, []) if legacy_strategy else [])
        """
        Step4: 构建选币因子的范围
        """
        factor_param_range = {}

        for factor_tuple in factor_tuple_list:
            factor_name = factor_tuple[0]

            if target_factors is None or factor_name in target_factors:
                factor_param_range[factor_name] = []
                for factor_val in self.factor_param_range_dict.get(factor_name, self.default_param_range):
                    factor_param = factor_tuple[:val_index] + (factor_val,) + factor_tuple[val_index + 1:]
                    factor_param_range[factor_name].append(factor_param)
            else:
                factor_param_range[factor_name] = [factor_tuple]
        if factor_param_range:
            return list(product(*factor_param_range.values()))
        else:
            return []

    def generate_combinations_by_strategy(self, strategy_dict: dict, target_factors: List[str] = None) -> List[dict]:
        """
        根据策略配置，和范围配置，生成指定策略的所有可能性
        :param strategy_dict: 策略配置
        :param target_factors: 指定的目标因子
        :return: 策略的所有可能性
        """
        """
        Step1: 把默认配置转成一个参数范围，范围内只有1个默认值。所有可能性只有1个，即原策略
        """
        strategy_param_range = {
            **{
                k: [v] for k, v in strategy_dict.items() if k != 'strategy'
            },  # 默认config转成list
        }
        """
        Step2: 构建选币因子的范围
        """
        for factor_attr in self.STRATEGY_FACTOR_ATTR:
            val_index = 1 if 'filter' in factor_attr else 2  # 格式不行一样
            candidates = self.get_candidates_by_factors(strategy_dict, factor_attr, target_factors, val_index)
            if candidates:
                strategy_param_range[factor_attr] = candidates
        """
        Step3: 更新默认配置
        - 情况1: 如果设置了 `target_factor`，我们只会循环遍历该因子。其余配置使用默认配置
        - 情况2: 如果没有设置，即需要循环所有可能性。所以我们会用策略配置好的范围，覆盖默认的范围（单参数模式）
        """
        # TODO: 目前框架暂不支持选配
        # if target_factor is None:
        #     strategy_param_range.update(self.strategy_param_range_dict)  # 如果定义了循环变量，会覆盖掉上面的默认设置
        """
        Step4: 根据策略的 strategy_param_range，生成所有的可能的配置
        """
        strategy_combinations = [
            dict(zip(strategy_param_range.keys(), combination), strategy=strategy_dict['strategy'])
            for combination in product(*strategy_param_range.values())
        ]
        return strategy_combinations

    def generate_configs(self, target_factors: list | tuple = None) -> List[BacktestConfig]:
        """
        根据配置的dict和默认的参数列表，自动生成所有的遍历参数组合
        :param target_factors: 可选变量，如果填写了的话，可以只针对这个变量遍历。其他的因子参数都使用策略默认值
        :return: BacktestConfig 的列表
        """
        """
        # Step1: 针对配置中每一个子策略，都生成所有的可能性，并且存档在 strategy_combinations_list 中
        """
        import config
        # 我们启用了大杂烩的支持，因此需要考虑多个子策略的情况
        strategy_combinations_list = []

        # 循环大杂烩中所有的子策略
        for strategy_dict in config.strategy_list:
            # 根据参数，生成单子策略的所有的组合
            strategy_combinations = self.generate_combinations_by_strategy(strategy_dict, target_factors)
            strategy_combinations_list.append(strategy_combinations)
        """
        # Step2: 把 strategy_combinations_list 转换为 我们要的策略组合模式
        把一个这样的数据结构
        `[[策略1的组合1, 策略1的组合2], [策略2的组合1, 策略2的组合2, 策略2的组合3]]`
        生成如下结果:
        [
          [策略1的组合1, 策略2的组合1], [策略1的组合1, 策略2的组合2], [策略1的组合1, 策略2的组合3],
          [策略1的组合2, 策略2的组合1], [策略1的组合2, 策略2的组合2], [策略1的组合2, 策略2的组合3]
        ]
        也就是 [strategy_list可能性1, strategy_list可能性2, ...]
        """
        strategy_list_combinations = list(product(*strategy_combinations_list))
        """
        # Step3: 根据所有可能的strategy list，生成所有的backtest_config
        """
        config_list: List[BacktestConfig] = []
        for index, strategy_list in enumerate(strategy_list_combinations):
            # 加载默认配置
            backtest_config = BacktestConfig.init_from_config(load_strategy_list=False)
            backtest_config.iter_round = index + 1
            # 使用指定的 strategy list 配置进行策略初始化
            backtest_config.load_strategy_config(strategy_list)
            if len(backtest_config.strategy_list) == 0:
                default_logger.critical('没有合法的策略，无法启动回测，跳过')
                continue

            self.update_meta_by_config(backtest_config)

            config_list.append(backtest_config)

        self.config_list = config_list

        return self.config_list

    def generate_long_and_short_configs(self) -> List[BacktestConfig]:
        """
        纯多/纯空的配置，用于多空曲线的计算
        :return:
        """
        import config

        long_short_strategy_list = []
        pure_long_strategy_list = []
        pure_short_strategy_list = []
        for strategy_dict in config.strategy_list:
            strategy_cfg = strategy_dict.copy()
            long_strategy_cfg = {**strategy_dict, **{'long_cap_weight': 1, 'short_cap_weight': 0}}
            short_strategy_cfg = {**strategy_dict, **{'long_cap_weight': 0, 'short_cap_weight': 1}}

            long_short_strategy_list.append(strategy_cfg)
            pure_long_strategy_list.append(long_strategy_cfg)
            pure_short_strategy_list.append(short_strategy_cfg)

        config_list: List[BacktestConfig] = []
        for stg, suffix in zip([long_short_strategy_list, pure_long_strategy_list, pure_short_strategy_list],
                               ['多空模拟', '纯多模拟', '纯空模拟']):
            backtest_config = BacktestConfig.init_from_config(load_strategy_list=False)
            backtest_config.load_strategy_config(stg)
            if len(backtest_config.strategy_list) == 0:
                default_logger.critical(f'【{suffix}场景】没有生成有效的子策略回测回测，可能所有选币都被重置为0，跳过')
                continue
            backtest_config.name = self.backtest_name
            backtest_config.iter_round = suffix

            self.update_meta_by_config(backtest_config)

            config_list.append(backtest_config)

        self.config_list = config_list

        return self.config_list

    def generate_configs_by_factor(self, *target_factors) -> List[BacktestConfig]:
        """
        生成单因子的配置，用于参数平原计算
        :param target_factors: 因子的名称
        :return:
        """
        return self.generate_configs(target_factors)

    def generate_all_factor_config(self):
        backtest_config = BacktestConfig.init_from_config(load_strategy_list=False)
        strategy_list = []
        for conf in self.config_list:
            strategy_list.extend(conf.strategy_list_raw)
        backtest_config.load_strategy_config(strategy_list)
        return backtest_config

    def get_name_params_sheet(self) -> pd.DataFrame:
        rows = []
        for config in self.config_list:
            rows.append(config.get_strategy_config_sheet())

        sheet = pd.DataFrame(rows)
        sheet.to_excel(self.config_list[-1].get_result_folder().parent / '策略回测参数总表.xlsx', index=False)
        return sheet

    def generate_configs_by_strategies(self, strategies, re_timing_strategies=None) -> List[BacktestConfig]:
        config_list = []
        iter_round = 0

        if not re_timing_strategies:
            re_timing_strategies = [None]

        for strategy_list, re_timing_config in product(strategies, re_timing_strategies):
            iter_round += 1
            backtest_config = BacktestConfig.init_from_config(load_strategy_list=False)
            if self.backtest_name:
                backtest_config.name = self.backtest_name
            backtest_config.load_strategy_config(strategy_list, re_timing_config)
            backtest_config.iter_round = iter_round

            self.update_meta_by_config(backtest_config)

            config_list.append(backtest_config)

        self.config_list = config_list

        return config_list


if __name__ == '__main__':
    for c in BacktestConfigFactory.init().generate_configs():
        print(c.get_fullname())
