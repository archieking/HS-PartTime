"""
选币策略回测框架，包含以下2个部分
1. 回测策略细节配置
2. 回测全局设置
"""
import os
from pathlib import Path

from core.utils.path_kit import get_folder_path, get_file_path

# ====================================================================================================
# ** 回测策略细节配置 **
# 需要配置需要的策略以及遍历的参数范围
# ====================================================================================================
# region 回测策略细节配置
start_date = '2021-01-01'  # 回测开始时间
end_date = '2024-10-05'  # 回测结束时间

backtest_name = '回测组'  # 回测组名称。可以自己任意取。一般建议，一个回测组，就是实盘中的一个账户。

"""策略配置"""
strategy_list = [
    {
        "strategy": "Strategy_Base",
        "offset_list": [0],
        "hold_period": "1H",
        "is_use_spot": True,
        "long_select_coin_num": 1,
        "short_select_coin_num": 1,
        "cap_weight": 1,
        "long_cap_weight": 1,  # 多头资金占比
        "short_cap_weight": 1,  # 空头资金占比
        "factor_list": [
            ('涨跌幅max', False, 100, 1)  # 因子名（和factors文件中相同），排序方式，参数，权重。
        ],
        "long_filter_list": [
            # ('涨跌幅max', 24, 'val:<=0.2'),  # 因子名（和factors文件中相同），参数 rank排名 val数值 pct百分比
            # ('VolumeMix', 24, 'pct:<=0.2', False)  # 因子名（和factors文件中相同），参数
        ],
        "short_filter_list": [
            # ('涨跌幅max', 24, 'val:<=0.2')  # 因子名（和factors文件中相同），参数
        ],
        "use_custom_func": False  # 使用系统内置因子计算、过滤函数
    }
    # for n in [_ for _ in range(90, 141, 10)]
]

factor_param_range_dict = {
    '涨跌幅max': [_ for _ in range(10, 201, 10)],
}

# 查看参数平原的图表类型
# - Y  每年参数平原
# - M  每月参数平原
# - Q  每季参数平原
params_plot_type = 'Y'

# 策略配置
min_kline_num = 168  # 最少上市多久，不满该K线根数的币剔除，即剔除刚刚上市的新币。168：标识168个小时，即：7*24
black_list = []  # 拉黑名单，永远不会交易。不喜欢的币、异常的币。例：LUNA-USDT, 这里与实盘不太一样，需要有'-'
# black_list = ['BTCUSDT', 'ETHUSDT', 'WBTCUSDT', 'WBETHUSDT', 'BNBUSDT', 'SOLUSDT']  # 拉黑名单，永远不会交易。不喜欢的币、异常的币。例：LUNA-USDT, 这里与实盘不太一样，需要有'-'
white_list = []  # 如果不为空，即只交易这些币，只在这些币当中进行选币。例：LUNA-USDT, 这里与实盘不太一样，需要有'-'

# 模拟下单回测设置
account_type = '普通账户'  # '统一账户'或者'普通账户'
initial_usdt = 10_000  # 初始资金
leverage = 1  # 杠杆数。
margin_rate = 0.05  # 维持保证金率，净值低于这个比例会爆仓

swap_c_rate = 8.5 / 10000  # 合约手续费(包含滑点)万5
spot_c_rate = 21 / 10000  # 现货手续费(包含滑点)千一

swap_min_order_limit = 5  # 合约最小下单量。最小不能低于5
spot_min_order_limit = 10  # 现货最小下单量。最小不能低于10

avg_price_col = 'avg_price_1m'  # 用于模拟计算的平均价，预处理数据使用的是1m，'avg_price_1m'表示1分钟的均价, 'avg_price_5m'表示5分钟的均价。
# endregion

# ====================================================================================================
# ** 回测全局设置 **
# 这些设置是客观事实，基本不会影响到回测的细节
# ====================================================================================================
# region 回测全局设置
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


raw_data_path = Path(get_folder_path('data'))
# 现货数据路径
spot_path = raw_data_path / 'spot_dict.pkl'
# 合约数据路径
swap_path = raw_data_path / 'swap_dict.pkl'

# 回测结果数据路径。用于发帖脚本使用
backtest_path = Path(get_folder_path('data', '回测结果'))
backtest_iter_path = Path(get_folder_path('data', '遍历结果'))

# 稳定币信息，不参与交易的币种
stable_symbol = ['BKRW', 'USDC', 'USDP', 'TUSD', 'BUSD', 'FDUSD', 'DAI', 'EUR', 'GBP', 'USBP', 'SUSD', 'PAXG', 'AEUR']
# endregion
