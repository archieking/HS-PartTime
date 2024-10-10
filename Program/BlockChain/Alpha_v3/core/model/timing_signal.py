import pandas as pd

from core.utils.signal_hub import SignalHub


class TimingSignal:

    def __init__(self, name: str, params: list | tuple = ()):
        self.name = name
        self.params = params

        signal_file = SignalHub.get_by_name(name)
        self.module_name = signal_file.module_name

        if hasattr(signal_file, 'signal'):
            self.signal = signal_file.signal

        if hasattr(signal_file, 'dynamic_leverage'):
            self.dynamic_leverage = signal_file.dynamic_leverage

    def signal(self, df, *args) -> pd.Series:
        """
        计算择时指标
        :param df: 原始行情数据
        :param args: 其他参数
        :return: 择时指标
        """
        raise NotImplementedError(f'请在`{self.module_name}`中实现 signal 方法')

    def dynamic_leverage(self, equity, *args) -> pd.Series:
        """
        根据资金曲线，动态调整杠杆
        :param equity: 资金曲线
        :param args: 其他参数
        :return: 返回包含 leverage 的数据
        """
        raise NotImplementedError(f'请在`{self.module_name}`中实现 dynamic_leverage 方法')

    def get_signal(self, df: pd.DataFrame):
        return self.signal(df, *self.params)

    def get_dynamic_leverage(self, equity: pd.Series):
        return self.dynamic_leverage(equity, *self.params)

    def __repr__(self) -> str:
        return f'{self.name}({self.params})'


if __name__ == '__main__':
    print(TimingSignal('MovingAverage', (5,)).get_signal(pd.DataFrame()))
