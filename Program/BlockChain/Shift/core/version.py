"""
仓位管理框架
"""
import time

from pandas import show_versions

from core.utils.log_kit import divider, logger
from core.utils.path_kit import get_file_path

sys_version = '1.0.0'
sys_name = 'position-management'
build_version = 'v1.0.0.20241112'


def version_prompt():
    show_versions()
    divider(f'版本: {sys_version}，当前时间:', '#', _logger=logger)
    logger.debug(f'BUILD VERSION: {build_version}')
