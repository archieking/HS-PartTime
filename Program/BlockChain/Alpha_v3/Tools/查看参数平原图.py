import pandas as pd

from core.figure import plotly_plot
from core.utils.path_kit import get_folder_path, get_file_path

csv_path = "/Users/archie/Projects/Quant/GitHub/Quant/Program/BCrypto/AlphaMain/AlphaMain_Backtest/Alpha_V3_Backtest/data/遍历结果/Swap_rcv_ak3_v2_zxp_L2S2_False_LF('涨跌幅max', 24, 'val:<=0.2')_SF('涨跌幅max', 24, 'val:<=0.2').csv"
res_name = csv_path.split('/')[-1].split('.')[0]

# df = pd.read_csv(csv_path, index_col='Unnamed: 0')
df = pd.read_csv(csv_path, index_col='index')
plotly_plot(df, get_folder_path('data', '参数平原'), f"历年参数平原图_{res_name}")
