import os
import sys
import json
import logging
from typing import Optional, Any, Dict


class Logger:
    _instance = None
    _logger = None

    def __init__(
        self,
        experiment_folder: Optional[str] = None,
        configs: Optional[Dict[str, Any]] = None,
        use_wandb: bool = False,
        wandb_name: Optional[str] = None,
        wandb_project: Optional[str] = None,
    ):
        self.experiment_folder = experiment_folder
        self.configs = configs or {}
        self.use_wandb = use_wandb
        self.wandb_name = wandb_name
        self.wandb_project = wandb_project
        self.metrics = {}

        self._logger = logging.getLogger("ThinkingPastTheAnswer")
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)

            if experiment_folder:
                os.makedirs(experiment_folder, exist_ok=True)
                fh = logging.FileHandler(os.path.join(experiment_folder, "run.log"), encoding="utf-8")
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                self._logger.addHandler(fh)

        Logger._instance = self
        Logger._logger = self._logger

    @classmethod
    def _get_raw_logger(cls):
        if cls._logger is None:
            raw = logging.getLogger("ThinkingPastTheAnswer")
            raw.setLevel(logging.INFO)
            if not raw.handlers:
                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.INFO)
                formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
                ch.setFormatter(formatter)
                raw.addHandler(ch)
            cls._logger = raw
        return cls._logger

    @classmethod
    def info(cls, msg: str):
        if cls._instance and cls._instance._logger:
            cls._instance._logger.info(msg)
        else:
            cls._get_raw_logger().info(msg)

    @classmethod
    def warning(cls, msg: str):
        if cls._instance and cls._instance._logger:
            cls._instance._logger.warning(msg)
        else:
            cls._get_raw_logger().warning(msg)

    @classmethod
    def error(cls, msg: str):
        if cls._instance and cls._instance._logger:
            cls._instance._logger.error(msg)
        else:
            cls._get_raw_logger().error(msg)

    @classmethod
    def debug(cls, msg: str):
        if cls._instance and cls._instance._logger:
            cls._instance._logger.debug(msg)
        else:
            cls._get_raw_logger().debug(msg)

    def log_metrics(self, metrics: Dict[str, Any]):
        self.metrics.update(metrics)
        self.info(f"Logged metrics: {metrics}")

    def show_metrics2table(self):
        if not self.metrics:
            self.info("No metrics recorded yet.")
            return
        header = f"{'Metric':<30} | {'Value':<15}"
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for k, v in self.metrics.items():
            lines.append(f"{str(k):<30} | {str(v):<15}")
        lines.append(sep)
        print("\n" + "\n".join(lines) + "\n")

    def save_metrics(self, filename: str = "results.json"):
        if not self.experiment_folder:
            return
        path = os.path.join(self.experiment_folder, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
        self.info(f"Metrics saved to {path}")

    def terminate_experiment(self):
        self.info("Experiment run finished.")
