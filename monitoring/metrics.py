"""
Prometheus指标收集模块
提供性能指标和业务指标收集
"""
import time
from typing import Dict, Any, Optional
from functools import wraps
from threading import Lock


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self._metrics: Dict[str, Any] = {}
        self._lock = Lock()
        self._start_time = time.time()
        
        # 初始化指标
        self._init_metrics()
    
    def _init_metrics(self):
        """初始化指标"""
        with self._lock:
            # HTTP请求指标
            self._metrics['http_requests_total'] = 0
            self._metrics['http_requests_by_method'] = {}
            self._metrics['http_requests_by_endpoint'] = {}
            self._metrics['http_request_duration_seconds'] = []
            
            # 特征提取指标
            self._metrics['feature_extractions_total'] = 0
            self._metrics['feature_extraction_duration_seconds'] = []
            self._metrics['feature_extraction_errors_total'] = 0
            
            # 检索指标
            self._metrics['retrieval_queries_total'] = 0
            self._metrics['retrieval_duration_seconds'] = []
            self._metrics['retrieval_results_count'] = []
            self._metrics['retrieval_errors_total'] = 0
            
            # 向量数据库指标
            self._metrics['vector_db_operations_total'] = 0
            self._metrics['vector_db_operations_by_type'] = {}
            self._metrics['vector_db_operation_duration_seconds'] = []
            self._metrics['vector_db_errors_total'] = 0
            
            # 系统指标
            self._metrics['system_uptime_seconds'] = 0
            self._metrics['active_connections'] = 0
    
    def record_http_request(self, method: str, endpoint: str, duration: float, status_code: int = 200):
        """记录HTTP请求指标"""
        with self._lock:
            self._metrics['http_requests_total'] += 1
            self._metrics['http_requests_by_method'][method] = \
                self._metrics['http_requests_by_method'].get(method, 0) + 1
            self._metrics['http_requests_by_endpoint'][endpoint] = \
                self._metrics['http_requests_by_endpoint'].get(endpoint, 0) + 1
            self._metrics['http_request_duration_seconds'].append(duration)
            # 只保留最近1000个请求的时长
            if len(self._metrics['http_request_duration_seconds']) > 1000:
                self._metrics['http_request_duration_seconds'] = \
                    self._metrics['http_request_duration_seconds'][-1000:]
    
    def record_feature_extraction(self, duration: float, success: bool = True):
        """记录特征提取指标"""
        with self._lock:
            self._metrics['feature_extractions_total'] += 1
            if success:
                self._metrics['feature_extraction_duration_seconds'].append(duration)
                if len(self._metrics['feature_extraction_duration_seconds']) > 1000:
                    self._metrics['feature_extraction_duration_seconds'] = \
                        self._metrics['feature_extraction_duration_seconds'][-1000:]
            else:
                self._metrics['feature_extraction_errors_total'] += 1
    
    def record_retrieval(self, duration: float, results_count: int, success: bool = True):
        """记录检索指标"""
        with self._lock:
            self._metrics['retrieval_queries_total'] += 1
            if success:
                self._metrics['retrieval_duration_seconds'].append(duration)
                self._metrics['retrieval_results_count'].append(results_count)
                if len(self._metrics['retrieval_duration_seconds']) > 1000:
                    self._metrics['retrieval_duration_seconds'] = \
                        self._metrics['retrieval_duration_seconds'][-1000:]
                if len(self._metrics['retrieval_results_count']) > 1000:
                    self._metrics['retrieval_results_count'] = \
                        self._metrics['retrieval_results_count'][-1000:]
            else:
                self._metrics['retrieval_errors_total'] += 1
    
    def record_vector_db_operation(self, operation_type: str, duration: float, success: bool = True):
        """记录向量数据库操作指标"""
        with self._lock:
            self._metrics['vector_db_operations_total'] += 1
            self._metrics['vector_db_operations_by_type'][operation_type] = \
                self._metrics['vector_db_operations_by_type'].get(operation_type, 0) + 1
            if success:
                self._metrics['vector_db_operation_duration_seconds'].append(duration)
                if len(self._metrics['vector_db_operation_duration_seconds']) > 1000:
                    self._metrics['vector_db_operation_duration_seconds'] = \
                        self._metrics['vector_db_operation_duration_seconds'][-1000:]
            else:
                self._metrics['vector_db_errors_total'] += 1
    
    def set_active_connections(self, count: int):
        """设置活跃连接数"""
        with self._lock:
            self._metrics['active_connections'] = count
    
    def _calculate_percentile(self, values: list, percentile: float) -> float:
        """计算百分位数（如p95）"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return float(sorted_values[min(index, len(sorted_values) - 1)])
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取所有指标 - 符合工程化要求：QPS、p95延迟、平均响应时间、GPU/CPU利用率"""
        with self._lock:
            # 更新运行时间
            self._metrics['system_uptime_seconds'] = time.time() - self._start_time
            
            # 计算平均值和百分位数
            metrics = self._metrics.copy()
            
            # HTTP请求指标
            if metrics['http_request_duration_seconds']:
                metrics['http_request_duration_seconds_avg'] = \
                    sum(metrics['http_request_duration_seconds']) / len(metrics['http_request_duration_seconds'])
                metrics['http_request_duration_seconds_p95'] = \
                    self._calculate_percentile(metrics['http_request_duration_seconds'], 95)
                metrics['http_request_duration_seconds_p99'] = \
                    self._calculate_percentile(metrics['http_request_duration_seconds'], 99)
                # 计算QPS（每秒请求数）
                uptime = metrics['system_uptime_seconds']
                metrics['http_requests_qps'] = metrics['http_requests_total'] / uptime if uptime > 0 else 0
            else:
                metrics['http_request_duration_seconds_avg'] = 0
                metrics['http_request_duration_seconds_p95'] = 0
                metrics['http_request_duration_seconds_p99'] = 0
                metrics['http_requests_qps'] = 0
            
            # 特征提取平均时长
            if metrics['feature_extraction_duration_seconds']:
                metrics['feature_extraction_duration_seconds_avg'] = \
                    sum(metrics['feature_extraction_duration_seconds']) / len(metrics['feature_extraction_duration_seconds'])
            else:
                metrics['feature_extraction_duration_seconds_avg'] = 0
            
            # 检索平均时长
            if metrics['retrieval_duration_seconds']:
                metrics['retrieval_duration_seconds_avg'] = \
                    sum(metrics['retrieval_duration_seconds']) / len(metrics['retrieval_duration_seconds'])
            else:
                metrics['retrieval_duration_seconds_avg'] = 0
            
            # 检索平均结果数
            if metrics['retrieval_results_count']:
                metrics['retrieval_results_count_avg'] = \
                    sum(metrics['retrieval_results_count']) / len(metrics['retrieval_results_count'])
            else:
                metrics['retrieval_results_count_avg'] = 0
            
            # 向量数据库操作平均时长
            if metrics['vector_db_operation_duration_seconds']:
                metrics['vector_db_operation_duration_seconds_avg'] = \
                    sum(metrics['vector_db_operation_duration_seconds']) / len(metrics['vector_db_operation_duration_seconds'])
            else:
                metrics['vector_db_operation_duration_seconds_avg'] = 0
            
            # GPU/CPU利用率（符合工程化要求）
            try:
                import torch
                import psutil
                
                # CPU利用率
                cpu_percent = psutil.cpu_percent(interval=0.1)
                metrics['cpu_utilization_percent'] = cpu_percent
                
                # GPU利用率（如果可用）
                if torch.cuda.is_available():
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    gpu_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    gpu_utilization = (gpu_allocated / gpu_total) * 100 if gpu_total > 0 else 0
                    metrics['gpu_utilization_percent'] = gpu_utilization
                    metrics['gpu_memory_total_gb'] = gpu_total
                    metrics['gpu_memory_allocated_gb'] = gpu_allocated
                else:
                    metrics['gpu_utilization_percent'] = 0
                    metrics['gpu_memory_total_gb'] = 0
                    metrics['gpu_memory_allocated_gb'] = 0
            except ImportError:
                metrics['cpu_utilization_percent'] = 0
                metrics['gpu_utilization_percent'] = 0
            except Exception:
                metrics['cpu_utilization_percent'] = 0
                metrics['gpu_utilization_percent'] = 0
            
            return metrics
    
    def get_prometheus_format(self) -> str:
        """获取Prometheus格式的指标"""
        metrics = self.get_metrics()
        lines = []
        
        # HTTP请求总数
        lines.append(f"http_requests_total {metrics['http_requests_total']}")
        
        # HTTP请求按方法分类
        for method, count in metrics['http_requests_by_method'].items():
            lines.append(f'http_requests_by_method{{method="{method}"}} {count}')
        
        # HTTP请求按端点分类
        for endpoint, count in metrics['http_requests_by_endpoint'].items():
            # 清理端点名称，移除特殊字符
            endpoint_clean = endpoint.replace('/', '_').replace('-', '_').replace('{', '').replace('}', '')
            lines.append(f'http_requests_by_endpoint{{endpoint="{endpoint_clean}"}} {count}')
        
        # HTTP请求平均时长和p95延迟（符合工程化要求）
        lines.append(f"http_request_duration_seconds_avg {metrics.get('http_request_duration_seconds_avg', 0)}")
        lines.append(f"http_request_duration_seconds_p95 {metrics.get('http_request_duration_seconds_p95', 0)}")
        lines.append(f"http_requests_qps {metrics.get('http_requests_qps', 0)}")
        
        # 特征提取指标
        lines.append(f"feature_extractions_total {metrics['feature_extractions_total']}")
        lines.append(f"feature_extraction_errors_total {metrics['feature_extraction_errors_total']}")
        lines.append(f"feature_extraction_duration_seconds_avg {metrics['feature_extraction_duration_seconds_avg']}")
        
        # 检索指标
        lines.append(f"retrieval_queries_total {metrics['retrieval_queries_total']}")
        lines.append(f"retrieval_errors_total {metrics['retrieval_errors_total']}")
        lines.append(f"retrieval_duration_seconds_avg {metrics['retrieval_duration_seconds_avg']}")
        lines.append(f"retrieval_results_count_avg {metrics['retrieval_results_count_avg']}")
        
        # 向量数据库指标
        lines.append(f"vector_db_operations_total {metrics['vector_db_operations_total']}")
        for op_type, count in metrics['vector_db_operations_by_type'].items():
            lines.append(f'vector_db_operations_by_type{{operation="{op_type}"}} {count}')
        lines.append(f"vector_db_errors_total {metrics['vector_db_errors_total']}")
        lines.append(f"vector_db_operation_duration_seconds_avg {metrics['vector_db_operation_duration_seconds_avg']}")
        
        # 系统指标（符合工程化要求：GPU/CPU利用率）
        lines.append(f"system_uptime_seconds {metrics['system_uptime_seconds']}")
        lines.append(f"cpu_utilization_percent {metrics.get('cpu_utilization_percent', 0)}")
        lines.append(f"gpu_utilization_percent {metrics.get('gpu_utilization_percent', 0)}")
        if metrics.get('gpu_memory_total_gb', 0) > 0:
            lines.append(f"gpu_memory_total_gb {metrics['gpu_memory_total_gb']}")
            lines.append(f"gpu_memory_allocated_gb {metrics.get('gpu_memory_allocated_gb', 0)}")
        lines.append(f"active_connections {metrics['active_connections']}")
        
        return "\n".join(lines)


# 全局指标收集器实例
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def measure_time(func):
    """装饰器：测量函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            # 可以在这里记录指标
            return result
        except Exception as e:
            duration = time.time() - start_time
            raise e
    return wrapper

