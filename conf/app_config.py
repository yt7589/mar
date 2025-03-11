#
import logging
from string import Formatter

class AppConfig(object):
    def __init__(self):
        self.name = 'conf.app_config.AppConfig'

    logger = logging.getLogger('Yantao')
    
    @staticmethod
    def initialize() -> None:
        AppConfig.logger.setLevel(logging.INFO)
        # 日志文件处理
        file_handler = logging.FileHandler('./work/logs/mar.log')
        file_handler.setLevel(logging.INFO)
        # 终端信息处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # 设置日志格式
        formmater = logging.Formatter('%(asctime)s  -  %(levelname)s  -  %(filename)s:%(lineno)d  -  %(funcName)s\n    %(message)s')
        file_handler.setFormatter(formmater)
        console_handler.setFormatter(formmater)
        # 设置当前日志为终端
        AppConfig.logger.addHandler(console_handler)