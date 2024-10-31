# 使用说明

## ☑️ 环境配置

为了确保程序在最新的 Python 3.12 环境下运行，我们需要使用 Anaconda 来创建一个新的环境，并安装所有依赖包。以下是详细的步骤：

### 1. 安装 Anaconda

如果您还没有安装 Anaconda，请先前往 [Anaconda 官方网站](https://www.anaconda.com/products/distribution) 下载并安装适用于您操作系统的
Anaconda 发行版。

### 2. 创建新环境

打开终端（Terminal）或 Anaconda Prompt，并执行以下命令来创建一个新的名为 `py312` 的环境，指定 Python 版本为 3.12：

```bash
conda create --name py312 python=3.12 -y
```

### 3. 激活环境

创建完成后，激活新环境：

```bash
conda activate py312
```

### 4. 安装依赖

确保您已经激活了 `py312` 环境，然后使用 conda 和 pip 安装 `requirements.txt` 文件中列出的所有依赖包。首先，使用 conda
安装一些常见的大型依赖包（如 pandas 和 matplotlib），然后使用 pip 安装剩余的依赖包。首先，确保您在项目的根目录下，然后执行以下命令：

#### 使用 conda 安装基础包

```bash
conda install pandas matplotlib -y
```

#### 使用 pip 安装其余依赖

如果您的网络连接不稳定，您可以使用国内的清华镜像源来加速依赖包的下载和安装。使用清华镜像源的命令如下：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

-----

## ▶️ 启动程序

安装所有依赖包后，您可以通过以下步骤启动程序：

### 1. 准备数据

下载预处理数据

解压缩之后把文件夹的路径在 [config.py](config.py)中进行配置 `raw_data_path`

⚠️ 注意：

- `raw_data_path` 需要是绝对路径，不要写成相对路径
- 路径文件夹下直接能看到所有的 pkl 文件，而不是子文件夹

### 2. 确认配置文件

确保您的配置文件 [config.py](config.py) 已正确设置，并放置在项目的根目录或指定的配置文件目录下。

### 3. 启动程序，一键回测

使用以下命令运行 `backtest.py` 来启动程序：

```bash
python backtest.py
```

### 4. 启动程序，一键回测多空曲线

使用以下命令运行 `backtest_multi_equity.py` 来启动程序：

```bash
python backtest_multi_equity.py
```

### 5. 启动程序，一键遍历参数平原

使用以下命令运行 `backtest_single_param.py` 来启动程序：

```bash
python backtest_single_param.py
```

### 5. 启动程序，一键遍历全参数

使用以下命令运行 `backtest_all_params.py` 来启动程序：

```bash
python backtest_all_params.py
```
-----

## 🗺️ 完整流程

以下是从创建环境到启动程序的完整命令列表：

```bash
# 安装 Anaconda 后，打开终端或 Anaconda Prompt
# 创建新环境
conda create --name py312 python=3.12 -y

# 激活环境
conda activate py312

# 使用 conda 安装基础包
conda install pandas matplotlib -y

# 使用 pip 安装其余依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 启动程序
python backtest.py
```

-----

## ℹ️ 常见问题

### 1. 环境无法激活

如果您遇到环境无法激活的问题，请确保 Anaconda 已正确安装，并且在终端或 Anaconda Prompt 中使用命令。

### 2. 依赖包安装失败

如果安装依赖包时遇到问题，请检查 `requirements.txt` 文件中的包名称和版本是否正确。如果网络连接不稳定，可以尝试使用国内镜像源来安装依赖包。

### 3. 程序启动失败

如果程序无法启动，请检查 `backtest.py` 文件和配置文件中的设置是否正确，确保所有路径和参数配置无误。

-----

## ⚙️ 初始化原理

### 策略初始化

1. config.py 中 strategy_list 为策略列表，逐一加载解析
2. 根据 `strategy` 配置的策略名，去 `strategy/` 目录下寻找对应的策略文件
3. 读取策略配置信息
    - 如果存在策略文件，加载策略文件
        1. 读取策略文件中所有的属性
        2. 使用config中所有的属性，
        3. 并更新策略文件中的对应属性
    - 如果策略文件不存在，直接读取config中所有的属性
4. 策略初始化
5. 加载函数
    - 如果配置了 `use_custom_func` 为 `True`，使用文件中自定义函数
    - 如果配置了 `use_custom_func` 为 `False`，使用内置函数