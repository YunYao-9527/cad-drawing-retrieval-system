"""
API网关层
提供REST API、异常捕获、日志记录
"""
from .app import create_app

__all__ = ['create_app']

