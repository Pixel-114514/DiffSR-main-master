import numpy as np
import torch
import yaml
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# -----------------------------------------------------------------------------
# 实验复现性与设备设置 (Reproducibility and Device Setup)
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    设置全局随机种子以确保实验的可复现性。

    Args:
        seed (int): 要设置的随机种子。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保卷积操作的确定性，但可能会牺牲一些性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_device(device_id: Optional[int] = None) -> torch.device:
    """
    根据可用性和用户选择设置计算设备。

    Args:
        device_id (Optional[int]): 指定的GPU设备ID。如果为None或CUDA不可用，
                                   则使用CPU。

    Returns:
        torch.device: 配置好的PyTorch设备对象。
    """
    if device_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

# -----------------------------------------------------------------------------
# 配置管理 (Configuration Management)
# -----------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    从YAML文件中安全地加载配置。

    Args:
        config_path (Path): 配置文件的路径对象。

    Returns:
        Dict[str, Any]: 包含配置信息的字典。
    """
    with open(config_path, 'r') as stream:
        # 使用 safe_load 替代 FullLoader，更安全
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return {}
    return config

def save_config(config: Dict[str, Any], save_dir: Path) -> None:
    """
    将配置字典保存到指定的目录下的config.yaml文件中。

    Args:
        config (Dict[str, Any]): 要保存的配置字典。
        save_dir (Path): 保存配置文件的目标目录。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False) # sort_keys=False 保持原始顺序

# -----------------------------------------------------------------------------
# 实验目录与日志设置 (Experiment Directory and Logging Setup)
# -----------------------------------------------------------------------------

def create_experiment_dir(
    base_path: str,
    model_name: str,
    dataset_name: str,
    add_timestamp: bool = True
) -> Path:
    """
    为实验创建一个带有时间戳的唯一目录。

    Args:
        base_path (str): 实验的根目录，例如 './experiments'。
        model_name (str): 模型名称。
        dataset_name (str): 数据集名称。
        add_timestamp (bool): 是否在目录名中添加日期和时间戳。

    Returns:
        Path: 创建的实验目录的路径对象。
    """
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}_{model_name}"
    else:
        dir_name = model_name

    # 使用 pathlib 拼接路径，更健壮
    exp_dir = Path(base_path) / dataset_name / dir_name
    
    # parents=True 递归创建父目录, exist_ok=True 目录存在时不报错
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def setup_logging(save_dir: Path) -> None:
    """
    配置日志系统，将日志同时输出到控制台和文件。

    Args:
        save_dir (Path): 保存日志文件的目录。
    """
    log_file_path = save_dir / "train.log"

    # 获取根 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除之前可能存在的 handlers，防止日志重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件 handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # 创建控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 定义日志格式
    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 将 handlers 添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Logging is set up. Log files will be saved in: {log_file_path}")