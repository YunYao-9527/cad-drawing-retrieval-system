"""
配置管理模块
提供集中式配置管理和环境变量支持
"""
from .config_manager import ConfigManager, get_config

__all__ = ['ConfigManager', 'get_config']

