import logging
import json
import sys
import torch
from collections import defaultdict
from typing import Union, List, Optional
from . import dist as dist_utils
from .utils.mics import get_device

class _ExperimentLogger:
    setup_count = 0

    def setup(
        self,
        logger_name: str = 'ml_experiment',
        file_log_level: int = logging.INFO,
        stream_log_level: int = logging.INFO,
        log_file: Optional[str] = 'experiment.log',
        formatter_str: str = '%(asctime)s - [%(name)s] - [%(levelname)s] - %(message)s'
    ):
        if self.setup_count > 0:
            raise RuntimeError("Logger 已经被初始化过了，请不要重复调用 setup 方法。")

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            formatter = logging.Formatter(formatter_str)

            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(stream_log_level)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            if log_file:
                file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                file_handler.setLevel(file_log_level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

        self._metrics = {}
        self.info(f"Logger '{logger_name}' 初始化成功。文件日志级别：{file_log_level}, 控制台日志级别：{stream_log_level}, 日志文件：{log_file if log_file else '无'}")
        _ExperimentLogger.setup_count += 1

    def debug(self, msg: str):
        """记录一条DEBUG级别的日志。"""
        self.logger.debug(msg)

    def info(self, msg: str):
        """记录一条INFO级别的日志。"""
        self.logger.info(msg)

    def warning(self, msg: str):
        """记录一条WARNING级别的日志。"""
        self.logger.warning(msg)

    def error(self, msg: str):
        """记录一条ERROR级别的日志。"""
        self.logger.error(msg)

    def critical(self, msg: str):
        """记录一条CRITICAL级别的日志。"""
        self.logger.critical(msg)


    def register_metric(self, name: str):
        if name in self._metrics:
            self.warning(f"指标 '{name}' 已存在，无需重复注册。")
        else:
            self._metrics[name] = []
            self.debug(f"指标 '{name}' 注册成功。")

    def record_metric(self, name: str, value):
        if name not in self._metrics:
            # self.warning(f"指标 '{name}' 未注册，将自动注册并记录数值。")
            self._metrics[name] = []

        self._metrics[name].append(value)

    def get_local_average(self, name: str) -> float:
        if name not in self._metrics:
            self.error(f"查询平均值失败: 指标 '{name}' 不存在。")
            return 0.0

        values = self._metrics[name]
        if not values:
            self.warning(f"指标 '{name}' 尚未记录任何数据，平均值为0.0。")
            return 0.0

        return sum(values) / len(values)

    def get_average(self, name: str) -> float:
        if not dist_utils.is_dist_avail_and_initialized():
            return self.get_local_average(name)

        local_avg = self.get_local_average(name)
        local_tensor = torch.tensor(local_avg, device=get_device())

        global_avg_tensor = dist_utils.reduce_tensor(local_tensor)
        return global_avg_tensor.item()

    def clear_metrics(self, names: Optional[List[str]] = None):
        if names is None:
            for name in self._metrics:
                self._metrics[name].clear()
        else:
            for name in names:
                if name in self._metrics:
                    self._metrics[name].clear()
                else:
                    self.warning(f"尝试清空失败: 指标 '{name}' 不存在。")

    def clear_metric(self, name: str):
        if name in self._metrics:
            self._metrics[name].clear()
        else:
            self.warning(f"尝试清空失败: 指标 '{name}' 不存在。")

    def get_all_metrics(self) -> dict:
        return self._metrics

    def save_metrics(self, filepath: str = 'metrics_summary.json'):
        self.info(f"准备将指标平均值摘要保存到 '{filepath}'...")

        average_metrics = {}

        for name in self._metrics.keys():
            average_value = self.get_average(name)
            average_metrics[name] = round(average_value, 6)

        self.info(f"计算出的平均值摘要: {average_metrics}")

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(average_metrics, f, ensure_ascii=False, indent=4)
            self.info(f"指标平均值摘要已成功保存到 '{filepath}'。")
        except IOError as e:
            self.error(f"保存指标摘要到 '{filepath}' 时发生IO错误: {e}")
        except Exception as e:
            self.error(f"保存指标摘要时发生未知错误: {e}")

mylogger = _ExperimentLogger()