import logging
import wandb
from time import time
from typing import Dict, Any
import torch
from utils.loss import LpLoss
from utils.metrics import Metrics
from datasets.ns2d import NavierStokes2DDataset  
from trainers import TRAINER_DICT, BaseTrainer 


def ns2d_procedure(config: Dict[str, Any]):
    """
    执行Navier-Stokes 2D 数据集的完整训练和评估流程。
    (函数文档保持不变)
    """
    model_name = config['train']['model_name']
    if model_name not in TRAINER_DICT:
        raise NotImplementedError(f"Model '{model_name}' not found in TRAINER_DICT.")

    is_main_process = not config.get('distributed', False) or config.get('rank') == 0

    if is_main_process and config['log'].get('wandb', False):
        run_name = f"{config['train']['model_name']}_{config['data']['dataset']}_{int(time())}"
        wandb.init(
            project=config['log'].get('wandb_project', 'default-project'), 
            name=run_name,
            config=config 
        )
    
    # --- 2. 加载数据集 ---
    # --- 修正: 从 'data' 子字典中获取数据集名称 ---
    logging.info(f"Loading dataset: {config['data']['dataset']}")
    start_time = time()
    # # 根据模型类型选择不同的数据集模块
    # if config['train']['model_type'] == 'diffusion':
    #     from datasets.ns2dsr import NavierStokes2DDataset
    # else:
    #     from datasets.ns2d import NavierStokes2DDataset
    
    dataset = NavierStokes2DDataset(
        distributed=config.get('distributed', False),
        **config['data'] 
    )
    
    train_loader = dataset.train_loader
    valid_loader = dataset.valid_loader
    test_loader = dataset.test_loader
    
    logging.info(f"Dataset loading completed in {time() - start_time:.2f}s")

    # --- 3. 构建Trainer和相关组件 ---
    logging.info(f"Building model trainer: {model_name}")
    start_time = time()
    
    # 实例化Trainer 
    trainer_class = TRAINER_DICT[model_name]
    trainer: BaseTrainer = trainer_class(config) 
    
    # --- 关键修正: 调用构建器时，传入对应的配置子字典 ---
    # model = trainer.build_model(config).to(config['train']['device'])
    # if config['distributed']:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=[config['train']['device']],  # 假设 device 是 torch.device 对象
    #         find_unused_parameters=True  # 防止某些参数未被用到导致 reduction 出错
    #     )
    model = trainer.build_model(config).to(config['train']['device'])
    optimizer = trainer.build_optimizer(model, config['optimize'])
    scheduler = trainer.build_scheduler(optimizer, config['schedule'])
    
    criterion = LpLoss(d=2, p=2, size_average=True)
    metrics = Metrics()
    
    if is_main_process: # 详细信息只在主进程打印
        logging.info(f"Model architecture:\n{model}")
        logging.info(f"Optimizer: {optimizer}")
        logging.info(f"Scheduler: {scheduler}")
        logging.info(f"Criterion: {criterion}")
    
    logging.info(f"Model and trainer setup completed in {time() - start_time:.2f}s")

    # --- 4. 启动训练流程 ---
    logging.info("Starting training process...")
    trainer.process(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        metrics=metrics,
    )
    logging.info("Training process finished.")