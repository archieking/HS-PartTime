"""
选币策略框架
"""
import time

from core.utils.log_kit import divider, logger
from core.utils.path_kit import get_file_path

sys_version = '1.2.1'
build_version = 'v1.2.1.20241009'


def version_prompt():
    divider(f'ACKNOWLEDGEMENT', '#', with_timestamp=False)
    acknowledgement = get_file_path('core', 'acknowledgement.md', as_path_type=True)
    print(acknowledgement.read_text(encoding='utf8').split('-----')[0].replace('\n\n', '\n').strip())
    divider(f'ACKNOWLEDGEMENT', '#', with_timestamp=False)
    time.sleep(1)
    print('\n\n')
    divider(f'版本: {sys_version}，当前时间:', '#', _logger=logger)
    logger.debug(f'BUILD VERSION: {build_version}')
