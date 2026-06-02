"""
健康检查模块
提供系统健康状态检查和指标导出
"""
import time
from typing import Dict, Any, Optional
from datetime import datetime

from config.config_manager import get_config
from monitoring.logger import get_logger, get_logger as _get_logger
from monitoring.metrics import get_metrics_collector
from services.feature_service import get_feature_service
from database.vector_db import get_vector_db


class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        """初始化健康检查器"""
        self.config = get_config()
        self.logger = _get_logger("health_check")
        self.metrics = get_metrics_collector()
        self.start_time = time.time()
        self.last_check_time = None
        self.last_check_status = None
    
    def check_health(self, detailed: bool = False) -> Dict[str, Any]:
        """
        执行健康检查
        
        Args:
            detailed: 是否执行详细检查
            
        Returns:
            健康状态字典
        """
        self.last_check_time = time.time()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "uptime_seconds": time.time() - self.start_time,
            "version": self.config.app.version
        }
        
        # 基本检查
        checks = {}
        
        # 检查特征服务
        try:
            feature_service = get_feature_service()
            model_info = feature_service.get_model_info()
            checks["feature_service"] = {
                "status": "healthy",
                "device": model_info.get("device"),
                "model_loaded": model_info.get("model_path") is not None
            }
        except Exception as e:
            checks["feature_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # 检查向量数据库
        try:
            vector_db = get_vector_db()
            stats = vector_db.get_stats()
            checks["vector_database"] = {
                "status": "healthy",
                "total_images": stats.get("total_images", 0),
                "db_type": stats.get("db_type", "unknown")
            }
        except Exception as e:
            checks["vector_database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # 详细检查
        if detailed:
            # 检查磁盘空间
            try:
                import shutil
                disk_usage = shutil.disk_usage(self.config.gallery.cad_drawing_dir)
                checks["disk_space"] = {
                    "status": "healthy",
                    "total_gb": disk_usage.total / (1024 ** 3),
                    "used_gb": disk_usage.used / (1024 ** 3),
                    "free_gb": disk_usage.free / (1024 ** 3),
                    "usage_percent": (disk_usage.used / disk_usage.total) * 100
                }
            except Exception as e:
                checks["disk_space"] = {
                    "status": "unknown",
                    "error": str(e)
                }
            
            # 检查GPU（如果可用）
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    gpu_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    checks["gpu"] = {
                        "status": "healthy",
                        "available": True,
                        "total_memory_gb": gpu_memory,
                        "allocated_memory_gb": gpu_allocated,
                        "free_memory_gb": gpu_memory - gpu_allocated
                    }
                else:
                    checks["gpu"] = {
                        "status": "healthy",
                        "available": False
                    }
            except Exception as e:
                checks["gpu"] = {
                    "status": "unknown",
                    "error": str(e)
                }
        
        health_status["checks"] = checks
        
        # 如果任何检查失败，更新状态
        for check_name, check_result in checks.items():
            if check_result.get("status") == "unhealthy":
                health_status["status"] = "unhealthy"
                break
        
        self.last_check_status = health_status["status"]
        
        return health_status
    
    def get_metrics(self) -> str:
        """
        获取Prometheus格式的指标
        
        Returns:
            Prometheus格式的指标字符串
        """
        return self.metrics.get_prometheus_format()
    
    def get_readiness(self) -> Dict[str, Any]:
        """
        获取就绪状态（用于Kubernetes readiness probe）
        
        Returns:
            就绪状态字典
        """
        health = self.check_health(detailed=False)
        
        return {
            "ready": health["status"] in ["healthy", "degraded"],
            "status": health["status"],
            "timestamp": health["timestamp"]
        }
    
    def get_liveness(self) -> Dict[str, Any]:
        """
        获取存活状态（用于Kubernetes liveness probe）
        
        Returns:
            存活状态字典
        """
        return {
            "alive": True,
            "uptime_seconds": time.time() - self.start_time,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# 全局健康检查器实例
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """获取全局健康检查器实例"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker

