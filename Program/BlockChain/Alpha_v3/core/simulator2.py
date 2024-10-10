"""
选币策略框架
"""

import numba as nb
import numpy as np
from numba.experimental import jitclass
"""
# 新语法说明
通过操作对象的值而不是更换reference，来保证所有引用的位置都能同步更新。

`self.target_lots[:] = target_lots`
这个写法涉及 Python 中的切片（slice）操作和对象的属性赋值。

`target_lots: nb.int64[:]  # 目标持仓手数`，self.target_lots 是一个列表，`[:]` 是切片操作符，表示对整个列表进行切片。

### 详细解释：

1. **`self.target_lots[:] = target_lots`**:
   - `self.target_lots` 是对象的一个属性，通常是一个列表（或者其它支持切片操作的可变序列）。
   - `[:]` 是切片操作符，表示对整个列表进行切片。具体来说，`[:]` 是对列表的所有元素进行选择，这种写法可以用于复制列表或对整个列表内容进行替换。

2. **具体操作**：
   - `self.target_lots[:] = target_lots` 不是直接将 `target_lots` 赋值给 `self.target_lots`，而是将 `target_lots` 中的所有元素替换 `self.target_lots` 中的所有元素。
   - 这种做法的一个好处是不会改变 `self.target_lots` 对象的引用，而是修改它的内容。这在有其他对象引用 `self.target_lots` 时非常有用，确保所有引用者看到的列表内容都被更新，而不会因为重新赋值而改变列表的引用。

### 举个例子：

```python
a = [1, 2, 3]
b = a
a[:] = [4, 5, 6]  # 只改变列表内容，不改变引用

print(a)  # 输出: [4, 5, 6]
print(b)  # 输出: [4, 5, 6]，因为 a 和 b 引用的是同一个列表，修改 a 的内容也影响了 b
```

如果直接用 `a = [4, 5, 6]` 替换 `[:]` 操作，那么 `b` 就不会受到影响，因为 `a` 重新指向了一个新的列表对象。
"""


@jitclass
class SwapMultiSimulator:
    """
    USDT 永续合约多标的回测模拟
    """
    margin_balance: float  # 账户权益, 单位 USDT
    comm_rate: float  # 手续费/交易成本
    min_order_limit: float  # 最小下单金额

    unrealized_pnl: nb.float64[:]  # 未实现盈亏

    lot_sizes: nb.float64[:]  # 每手币数，表示一手加密货币中包含的币数
    lots: nb.int64[:]  # 当前持仓手数

    last_prices: nb.float64[:]  # 最新价格
    has_last_prices: bool  # 是否有最新价

    def __init__(self, init_capital, lot_sizes, comm_rate, init_lots, min_order_limit):
        self.margin_balance = init_capital  # 账户权益
        self.comm_rate = comm_rate  # 交易成本
        self.min_order_limit = min_order_limit  # 最小下单金额

        n = len(lot_sizes)

        self.unrealized_pnl = np.zeros(n, dtype=np.float64)

        # 合约面值
        self.lot_sizes = np.zeros(n, dtype=np.float64)
        self.lot_sizes[:] = lot_sizes

        # 前收盘价
        self.last_prices = np.zeros(n, dtype=np.float64)
        self.has_last_prices = False

        # 当前持仓手数
        self.lots = np.zeros(n, dtype=np.int64)
        self.lots[:] = init_lots

    def fill_last_prices(self, prices):
        mask = np.logical_not(np.isnan(prices))
        self.last_prices[mask] = prices[mask]
        self.has_last_prices = True

    def settle_equity(self, prices):
        """
        结算当前账户权益
        :param prices: 当前价格
        :return:
        """
        mask = np.logical_and(self.lots != 0, np.logical_not(np.isnan(prices)))
        pnl_delta = np.zeros(len(self.lot_sizes), dtype=np.float64)

        # 盈亏变化 = (最新价格 - 前最新价（前收盘价）) * 持币数量。其中，持币数量 = min_qty * 持仓手数。
        pnl_delta[mask] = (prices[mask] - self.last_prices[mask]) * self.lot_sizes[mask] * self.lots[mask]

        # 反映到各 symbol 未实现盈亏上
        self.unrealized_pnl += pnl_delta

        # 总盈亏变化 = 所有币种对应的盈亏变化累加起来
        pnl_delta_total = np.sum(pnl_delta)

        # 反映到账户总权益上
        self.margin_balance += pnl_delta_total

    def on_open(self, open_prices, funding_rates, mark_prices):
        """
        模拟: K 线开盘 -> K 线收盘时刻
        :param open_prices: 开盘价
        :param funding_rates: 资金费
        :param mark_prices: 计算资金费的标记价格（目前就用开盘价来）
        :return:
        """
        if not self.has_last_prices:
            self.fill_last_prices(open_prices)

        # 根据开盘价和前最新价（前收盘价），结算当前账户权益
        self.settle_equity(open_prices)

        # 根据标记价格和资金费率，结算资金费盈亏
        mask = np.logical_and(self.lots != 0, np.logical_not(np.isnan(mark_prices)))
        notional_value = self.lot_sizes[mask] * self.lots[mask] * mark_prices[mask]
        funding_fee = np.sum(notional_value * funding_rates[mask])
        self.margin_balance -= funding_fee

        # 最新价为开盘价
        self.fill_last_prices(open_prices)

        # 返回扣除资金费后开盘账户权益、资金费和带方向的仓位名义价值
        return self.margin_balance, funding_fee, np.sum(np.abs(notional_value)), np.sum(self.unrealized_pnl)

    def on_execution(self, target_lots, exec_prices):
        if not self.has_last_prices:
            self.fill_last_prices(exec_prices)

        # 模拟: K 线开盘时刻 -> 调仓时刻

        # 根据调仓价和前最新价（开盘价），结算当前账户权益
        self.settle_equity(exec_prices)

        # 计算需要买入或卖出的合约数量
        delta = target_lots - self.lots
        mask = np.logical_and(delta != 0, np.logical_not(np.isnan(exec_prices)))

        # 计算成交额
        turnover_abs = np.zeros(len(self.lot_sizes), dtype=np.float64)
        turnover_abs[mask] = np.abs(delta[mask]) * self.lot_sizes[mask] * exec_prices[mask]

        # 成交额小于 min_order_limit 则无法调仓
        mask = np.logical_and(mask, turnover_abs >= self.min_order_limit)

        delta[~mask] = 0
        turnover_abs[~mask] = 0

        # 本期调仓总成交额
        turnover_abs_total = turnover_abs.sum()

        if np.isnan(turnover_abs_total):
            raise RuntimeError('Turnover is nan')

        # 根据总成交额计算并扣除手续费
        fee = turnover_abs_total * self.comm_rate
        self.margin_balance -= fee

        # 平仓数量
        close_amount = np.zeros(len(self.lot_sizes), dtype=np.int64)

        # 持仓方向和数量
        lots_dire, lots_amount = np.sign(self.lots), np.abs(self.lots)

        # 仓位变化方向和数量
        delta_dire, delta_amount = np.sign(delta), np.abs(delta)

        # 如果有持仓，且和仓位变化方向相反，乘积为负数，则涉及平仓
        mask_close = (lots_dire * delta_dire) < 0
        # 反之:
        # Case 1.1: 无仓位变化，乘积为0，不涉及平仓
        # Case 1.2: 无持仓且有仓位变化，乘积为0，为开仓，不涉及平仓
        # Case 2: 持仓与仓位变化方向相同，乘积为正，为开仓，不涉及平仓

        # 平仓数量 = min(持仓数量, 仓位变化数量)，如果变化量更多，则平完了还要开反向仓位
        close_amount[mask_close] = np.minimum(lots_amount[mask_close], delta_amount[mask_close])

        # 完全平仓
        mask_close_all = np.logical_and(mask_close, close_amount == lots_amount)

        # 完全平仓则未实现盈亏归零
        self.unrealized_pnl[mask_close_all] = 0

        # 部分平仓
        mask_close_partial = np.logical_and(mask_close, close_amount < lots_amount)

        # 部分平仓则按平仓数量占比结算未实现盈亏
        realize_ratio = close_amount[mask_close_partial].astype(np.float64) / lots_amount[mask_close_partial]
        self.unrealized_pnl[mask_close_partial] *= 1.0 - realize_ratio

        # 更新已成功调仓的 symbol 持仓
        self.lots[mask] = target_lots[mask]

        # 最新价为调仓价
        self.fill_last_prices(exec_prices)

        # 返回扣除手续费的调仓后账户权益，成交额，和手续费
        return self.margin_balance, turnover_abs_total, fee

    def on_close(self, close_prices):
        if not self.has_last_prices:
            self.fill_last_prices(close_prices)

        # 模拟: 调仓时刻 -> K 线收盘时刻

        # 根据收盘价和前最新价（调仓价），结算当前账户权益
        self.settle_equity(close_prices)

        # 最新价为收盘价
        self.fill_last_prices(close_prices)

        # 返回收盘账户权益
        return self.margin_balance


@jitclass
class SpotMultiSimulator:
    """
    现货多标的回测模拟
    """
    usdt_balance: float  # 账户 USDT 余额
    comm_rate: float  # 手续费/交易成本
    min_order_limit: float  # 最小下单金额

    lot_sizes: nb.float64[:]  # 每手币数，表示一手加密货币中包含的币数
    lots: nb.int64[:]  # 当前持仓手数

    def __init__(self, init_usdt_balance, lot_sizes, comm_rate, init_lots, min_order_limit):
        self.usdt_balance = init_usdt_balance  # 账户 USDT 余额
        self.comm_rate = comm_rate  # 交易成本
        self.min_order_limit = min_order_limit  # 最小下单金额

        n = len(lot_sizes)

        # 每手币数
        self.lot_sizes = np.zeros(n, dtype=np.float64)
        self.lot_sizes[:] = lot_sizes

        # 初始持币手数
        self.lots = np.zeros(n, dtype=np.int64)
        self.lots[:] = init_lots

    def get_asset_usdt_value(self, prices):
        """
        计算当前持币 USDT 价值
        不考虑融币卖空
        """
        mask = np.logical_and(self.lots != 0, np.logical_not(np.isnan(prices)))

        # 持币 USDT 价值
        asset_usdt_values = self.lots[mask] * self.lot_sizes[mask] * prices[mask]
        return np.sum(asset_usdt_values)

    def get_account_usdt_equity(self, prices):
        """
        计算现货账户总权益，以 USDT 计价
        可以融 U 做多，不考虑融币卖空
        """
        asset_usdt_value = self.get_asset_usdt_value(prices)

        # 总权益 = USDT 余额 + 持币 USDT 价值总和
        usdt_equity = self.usdt_balance + self.get_asset_usdt_value(prices)
        return asset_usdt_value, usdt_equity

    def on_execution(self, target_lots, exec_prices):
        """
        模拟: 现货调仓
        """

        # 计算需要买入或卖出的现货手数
        delta = target_lots - self.lots
        mask = np.logical_and(delta != 0, np.logical_not(np.isnan(exec_prices)))

        # 计算带买卖方向的成交额，正成交额代表买入，负成交额代表卖出
        turnover = np.zeros(len(self.lot_sizes), dtype=np.float64)
        turnover[mask] = delta[mask] * self.lot_sizes[mask] * exec_prices[mask]

        # 成交额小于 min_order_limit 则无法调仓
        mask = np.logical_and(mask, np.abs(turnover) >= self.min_order_limit)

        # 清除无法调仓的成交额
        turnover[~mask] = 0

        # 本期调仓总成交额
        turnover_abs_total = np.abs(turnover[mask]).sum()

        if np.isnan(turnover_abs_total):
            raise RuntimeError('Turnover is nan')

        # 根据总成交额计算并扣除手续费
        fee = turnover_abs_total * self.comm_rate
        self.usdt_balance -= fee

        # 更新已成功调仓的 symbol 持仓
        self.lots[mask] = target_lots[mask]

        # 结算当前 USDT 余额
        self.usdt_balance -= np.sum(turnover)

        # 返回成交额和手续费
        return turnover_abs_total, fee
