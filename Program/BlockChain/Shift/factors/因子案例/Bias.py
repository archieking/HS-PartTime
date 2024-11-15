"""
仓位管理框架
"""
import pandas as pd


def signal(df: pd.DataFrame, *args) -> pd.DataFrame:
    n = int(args[0])
    factor_name = args[1]

    ma = df['close'].rolling(n, min_periods=1).mean()
    bias = df['close'] / ma
    df[factor_name] = bias.pct_change(n)

    return df


def rotation(df: pd.DataFrame, *args) -> pd.DataFrame:
    """
    根据资金曲线，计算轮动因子
    :param df: 包含资金曲线的 DataFrame
    :param args: 其他参数
    :return: 包含轮动因子的 DataFrame
    """

    # ===== 获取策略参数
    n = int(args[0])
    factor_name = args[1]

    # ===== 计算指标
    # 归一化作为 close 计算因子
    df['close'] = (df['equity'] / df['equity'].iloc[0])
    df = signal(df, n, factor_name)
    df.drop(columns='close', inplace=True)

    return df


def dynamic_leverage(equity: pd.Series, *args) -> pd.Series:
    """
    根据资金曲线，动态调整杠杆
    :param equity: 资金曲线
    :param args: 其他参数
    :return: 返回包含 leverage 的数据
    """

    # ===== 获取策略参数
    n = int(args[0])

    # ===== 计算指标
    # 归一化作为 close 计算因子
    df = (equity / equity.iloc[0]).to_frame('close')
    factor_name = f'bias_{n}'

    df = signal(df, n, factor_name)

    # 默认空仓
    leverage = pd.Series(0., index=equity.index)

    # bias > 0，才持有
    above = df[factor_name] > 0
    leverage.loc[above] = 1.

    return leverage
