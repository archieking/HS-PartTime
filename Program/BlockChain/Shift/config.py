"""
选币策略回测框架，包含以下2个部分
1. 回测策略细节配置
2. 回测全局设置
"""
import os
from pathlib import Path

from core.utils.path_kit import get_folder_path

# ====================================================================================================
# ** 回测策略细节配置 **
# ====================================================================================================
# region 回测策略细节配置
start_date = '2020-01-01'  # 回测开始时间
end_date = '2024-11-01'  # 回测结束时间

# 数据存储路径，填写绝对路径
pre_data_path = r'/Users/archie/Projects/Quant/GitHub/Quant/Program/BCrypto/AlphaMain/AlphaMain_Backtest/Alpha_V3_Backtest/data'

backtest_name = '因子轮动'  # 回测的策略组合的名称。可以自己任意取。一般建议，一个回测组，就是实盘中的一个账户。

pos_strategy_config = {
    'name': 'RotationStrategy',  # *必填。使用什么策略，这里是轮动策略
    'hold_period': '1H',  # *必填。聚合后策略持仓周期。目前回测支持日线级别、小时级别。例：1H，6H，3D，7D......
    'params': {  # 非必填。聚合类策略的参数，这里的案例是在轮动策略场景下，我们需要提供轮动因子。
        'factor_list': [('Bias', False, 168, 1)],
    }
}

pos_strategy_candidates = [
    # 这里的选币策略和选币框架的策略配置完全相同，可以配置多空分离等复杂选币策略
    dict(
        name='轮动策略1',
        strategy_list=[
            {
                "strategy": "Strategy_LongShort",
                "offset_list": list(range(24)),
                "hold_period": "24H",
                "is_use_spot": True,
                "cap_weight": 1,
                "factor_list": [('QuoteVolumeMean', True, 96, 1)],
                "filter_list": [('PctChange', 96, 'pct:<0.8')],
                "use_custom_func": False
            },
        ],
    ),
    # 这里的选币策略和选币框架的策略配置完全相同，可以配置多空分离等复杂选币策略
    dict(
        name='轮动策略2',
        strategy_list=[
            {
                "strategy": "Strategy_LongOnly",
                "offset_list": list(range(24)),
                "hold_period": "24H",
                "is_use_spot": True,
                "cap_weight": 1,
                "long_cap_weight": 1,  # 策略内多头资金权重
                "short_cap_weight": 0,  # 策略内空头资金权重
                "factor_list": [('QuoteVolumeMean', True, 168, 1)],
                "filter_list": [('PctChange', 168, 'pct:<0.8')],
                "use_custom_func": False
            },
        ],
    ),
    # 可以继续配置第3个或者更多想要的策略
]

# 策略配置
min_kline_num = 168  # 最少上市多久，不满该K线根数的币剔除，即剔除刚刚上市的新币。168：标识168个小时，即：7*24
black_list = []  # 拉黑名单，永远不会交易。不喜欢的币、异常的币。例：LUNA-USDT, 这里与实盘不太一样，需要有'-'
# black_list = ['BTC-USDT', 'ETH-USDT']  # 拉黑名单，永远不会交易。不喜欢的币、异常的币。例：LUNA-USDT, 这里与实盘不太一样，需要有'-'
white_list = []  # 如果不为空，即只交易这些币，只在这些币当中进行选币。例：LUNA-USDT, 这里与实盘不太一样，需要有'-'

# 模拟下单回测设置
account_type = '普通账户'  # '统一账户'或者'普通账户'
initial_usdt = 1_0000  # 初始资金
leverage = 1  # 杠杆数。我看哪个赌狗要把这里改成大于1的。高杠杆如梦幻泡影。不要想着一夜暴富，脚踏实地赚自己该赚的钱。
margin_rate = 0.05  # 维持保证金率，净值低于这个比例会爆仓

swap_c_rate = 8.5 / 10000  # 合约手续费(包含滑点)万5
spot_c_rate = 21 / 10000  # 现货手续费(包含滑点)千一

swap_min_order_limit = 5  # 合约最小下单量。最小不能低于5
spot_min_order_limit = 10  # 现货最小下单量。最小不能低于10

avg_price_col = 'avg_price_1m'  # 用于模拟计算的平均价，预处理数据使用的是1m，'avg_price_1m'表示1分钟的均价, 'avg_price_5m'表示5分钟的均价。

reserved_cache = ['select']  # 用于缓存控制：['select']表示只缓存选币结果，不缓存其他数据，['all']表示缓存所有数据。
# 目前支持选项：
# - select: 选币结果pkl
# - strategy: 大杂烩中策略选币pkl
# - ratio: 最终模拟持仓的各个币种资金占比
# - all: 无视上述配置细节，包含 `all` 就代表我全要
# 缓存东西越多，硬盘消耗越大，对于参数比较多硬盘没那么大的童鞋，可以在这边设置
# endregion

# ====================================================================================================
# ** 回测全局设置 **
# 这些设置是客观事实，基本不会影响到回测的细节
# ====================================================================================================
job_num = os.cpu_count() - 2  # 回测并行数量

# ==== factor_col_limit 介绍 ====
factor_col_limit = 64  # 内存优化选项，一次性计算多少列因子。64是 16GB内存 电脑的典型值
# - 数字越大，计算速度越快，但同时内存占用也会增加。
# - 该数字是在 "因子数量 * 参数数量" 的基础上进行优化的。
#   - 例如，当你遍历 200 个因子，每个因子有 10 个参数，总共生成 2000 列因子。
#   - 如果 `factor_col_limit` 设置为 64，则计算会拆分为 ceil(2000 / 64) = 32 个批次，每次最多处理 64 列因子。
# - 对于16GB内存的电脑，在跑含现货的策略时，64是一个合适的设置。
# - 如果是在16GB内存下跑纯合约策略，则可以考虑将其提升到 128，毕竟数值越高计算速度越快。
# - 以上数据仅供参考，具体值会根据机器配置、策略复杂性、回测周期等有所不同。建议大家根据实际情况，逐步测试自己机器的性能极限，找到适合的最优值。

# 路径处理
raw_data_path = Path(get_folder_path(pre_data_path))  # 预处理数据路径
spot_path = raw_data_path / 'spot_dict.pkl'  # 现货数据路径
swap_path = raw_data_path / 'swap_dict.pkl'  # 合约数据路径

# 回测结果数据路径。用于发帖脚本使用
backtest_path = Path(get_folder_path('data', '回测结果'))
backtest_iter_path = Path(get_folder_path('data', '遍历结果'))

# 稳定币信息，不参与交易的币种
stable_symbol = ['BKRW', 'USDC', 'USDP', 'TUSD', 'BUSD', 'FDUSD', 'DAI', 'EUR', 'GBP', 'USBP', 'SUSD', 'PAXG', 'AEUR']

if spot_path.exists() is False or swap_path.exists() is False:
    print('⚠️ 请先准确配置预处理数据的位置（pre_data_path）。建议直接复制绝对路径，并且粘贴给 pre_data_path')
    exit()
