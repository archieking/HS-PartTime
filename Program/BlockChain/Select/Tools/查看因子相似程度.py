import os
import difflib
import re

def read_code_file(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    # 去除注释、空格和换行
    code = re.sub(r'#.*', '', code)  # 去除单行注释
    code = re.sub(r'\'\'\'.*?\'\'\'', '', code, flags=re.DOTALL)  # 去除多行注释
    code = re.sub(r'\s+', '', code)  # 去除空格和换行
    return code

def calculate_similarity(code1, code2):
    return difflib.SequenceMatcher(None, code1, code2).ratio()

def get_code_files(directory):
    code_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                code_files.append(os.path.join(root, file))
    return code_files

def main(directory):
    code_files = get_code_files(directory)
    similarities = {}
    for i, file1 in enumerate(code_files):
        for j, file2 in enumerate(code_files):
            if i < j:
                code1 = read_code_file(file1)
                code2 = read_code_file(file2)
                similarity = calculate_similarity(code1, code2)
                similarities[(file1, file2)] = similarity
    return similarities

# 示例用法
directory = '/Users/archie/Projects/Quant/GitHub/Quant/Program/BCrypto/AlphaMain/AlphaMain_Backtest/Alpha_V3_Backtest/factors/True因子/Swap/S12'
similarities = main(directory)
for (file1, file2), similarity in similarities.items():
    print(f'Similarity between: \n{file1} \n{file2} \n{similarity:.2f}\n')