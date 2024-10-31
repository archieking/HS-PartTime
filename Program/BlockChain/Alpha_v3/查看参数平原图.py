import pandas as pd

from core.figure import plotly_plot
from core.utils.path_kit import get_folder_path, get_file_path

csv_path = "/Users/moon/HangSingQuant/Program/BlockChain/Alpha_v3/data/遍历结果/Spot_Bias-v10-xqq_L31S31_True_Y.csv"
res_name = csv_path.split('/')[-1].split('.')[0]

df = pd.read_csv(csv_path, index_col='Unnamed: 0')
# df = pd.read_csv(csv_path, index_col='index')
plotly_plot(df, get_folder_path('data', '参数平原'), f"历年参数平原图_{res_name}")
