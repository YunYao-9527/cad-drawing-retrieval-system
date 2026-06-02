"""
结构化日志模块
支持JSON和文本格式日志输出
"""
import logging
import json
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON格式日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        return json.dumps(log_data, ensure_ascii=False)


class StructuredFormatter(logging.Formatter):
    """结构化文本格式日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为结构化文本"""
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        log_parts = [
            f"[{timestamp}]",
            f"[{record.levelname:8s}]",
            f"[{record.name}]",
            f"{record.getMessage()}"
        ]
        
        if record.exc_info:
            log_parts.append(f"\n{self.formatException(record.exc_info)}")
        
        return " ".join(log_parts)


def setup_logger(
    name: str = "cad_retrieval",
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    max_bytes: int = 10485760,
    backup_count: int = 5,
    enable_console: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: 日志格式 (json, text)
        log_file: 日志文件路径
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的备份文件数量
        enable_console: 是否启用控制台输出
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 选择格式化器
    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = StructuredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # 控制台处理器
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "cad_retrieval") -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器实例
    """
    return logging.getLogger(name)

