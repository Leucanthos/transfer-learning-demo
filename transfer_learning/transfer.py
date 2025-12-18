import pickle
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import lightgbm as lgb

from .base import ModelBase

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_booster(model_path: str) -> lgb.Booster:
    """加载pickle格式的LightGBM模型"""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Source model not found at {model_path}")
    with path.open("rb") as f:
        return pickle.load(f)


def pseudo_label_transfer_learning(
    source_data: pd.DataFrame,
    target_data: pd.DataFrame,
    test_data: pd.DataFrame,
    x_cols: list,
    y_cols: list,
    source_model_path: str,
    pseudo_model_path: Optional[str] = None,
    weight_col: str = "day_gap",
    valid_days: int = 7,
    device: str = "cpu",
    seed: int = 2018,
) -> Dict[str, Any]:
    """
    伪标签迁移学习方法
    
    使用源域模型为为目标域数据生成伪标签，然后将带有伪标签的目标域数据
    与源域数据一起训练新模型
    
    参数:
    source_data: 源域训练数据 (AIR)
    target_data: 目标域训练数据 (HPG)
    test_data: 测试数据
    x_cols: 特征列名列表
    y_cols: 目标列名列表
    source_model_path: 源模型路径
    pseudo_model_path: 保存模型路径
    weight_col: 权重列名
    valid_days: 验证天数
    device: 训练设备 ("cpu" 或 "gpu")
    seed: 随机种子
    
    返回:
    包含预测结果和RMSLE评分的字典
    """
    model_base = ModelBase(device=device)
    
    # 加载源模型
    model_base = ModelBase(device=device)
    
    # Prepare datasets
    tr_x, tr_y = target_data[x_cols], target_data[y_cols].values.ravel()
    te_x, te_y = test_data[x_cols], test_data[y_cols].values.ravel()
    
    # Generate pseudo labels for target data using source model
    source_model = _load_booster(source_model_path)
    pseudo_labels = source_model.predict(tr_x)
    
    # Combine source data with pseudo-labeled target data
    # Note: We're assuming x_cols and y_cols are consistent between source and target
    combined_x = pd.concat([source_data[x_cols], tr_x], ignore_index=True)
    combined_y = np.concatenate([source_data[y_cols].values.ravel(), pseudo_labels])
    
    # Apply weights based on recency (more recent = higher weight)
    tr_weight = 1 / (1 + np.abs(target_data[weight_col]))
    pseudo_weight = np.ones(len(source_data))  # Equal weight for source data
    
    # Normalize weights
    tr_weight = tr_weight / tr_weight.mean()
    pseudo_weight = pseudo_weight / pseudo_weight.mean()
    combined_weight = np.concatenate([pseudo_weight, tr_weight])
    
    # Create datasets
    train_matrix = lgb.Dataset(combined_x, combined_y, weight=combined_weight, free_raw_data=False)
    valid_matrix = lgb.Dataset(te_x, te_y, free_raw_data=False)
    
    # Train model
    model = model_base.train_or_load_lgbm(
        train_matrix,
        valid_matrix,
        pseudo_model_path,
        num_threads=4,
        seed=seed,
    )
    
    # Evaluate
    predictions = model.predict(te_x, num_iteration=model.best_iteration)
    rmsle_score = np.sqrt(mean_squared_error(te_y, predictions))
    
    return {
        "model": model,
        "predictions": predictions,
        "actual": te_y,
        "rmsle": rmsle_score,
    }


def fine_tune_transfer_learning(
    source_data: pd.DataFrame,
    target_data: pd.DataFrame,
    test_data: pd.DataFrame,
    x_cols: List[str],
    y_cols: List[str],
    source_model_path: str,
    fine_tune_lr: float = 0.01,
    weight_col: str = "day_gap",
    valid_days: int = 7,
    num_round: int = 1000,
    device: str = "cpu",
    seed: int = 2018,
) -> Dict[str, Any]:
    """
    微调迁移学习方法
    
    加载预训练的源模型并在目标域数据上进行微调
    
    参数:
        source_data: 源域训练数据 (AIR)
        target_data: 目标域训练数据 (HPG)
        test_data: 测试数据
        x_cols: 特征列名列表
        y_cols: 目标列名列表
        source_model_path: 源模型路径
        fine_tune_lr: 微调学习率
        weight_col: 权重列名
        valid_days: 验证天数
        num_round: 训练轮数
        device: 计算设备 ("cpu" 或 "gpu")
        seed: 随机种子
        
    返回:
        包含模型、预测结果、实际值和RMSLE分数的字典
    """
    model_base = ModelBase(device=device)
    
    # Prepare datasets
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=target_data,
        test_data=test_data,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        valid_days=valid_days,
    )
    
    # Load source model for initialization
    source_model = _load_booster(source_model_path)
    
    # 在目标域数据上微调模型
    model = model_base.train_or_load_lgbm(
        train_matrix,
        valid_matrix,
        None,  # 不保存中间模型
        learning_rate=fine_tune_lr,
        num_round=num_round,
        num_threads=4,
        seed=seed,
        early_stopping_rounds=300,
        init_model=source_model,
    )
    
    # Evaluate
    predictions = model.predict(te_x, num_iteration=model.best_iteration)
    rmsle_score = np.sqrt(mean_squared_error(te_y, predictions))
    
    return {
        "model": model,
        "predictions": predictions,
        "actual": te_y,
        "rmsle": rmsle_score,
    }


def direct_transfer_with_sample_selection(
    source_data: pd.DataFrame,
    target_data: pd.DataFrame,
    test_data: pd.DataFrame,
    x_cols: List[str],
    y_cols: List[str],
    source_model_path: str,
    sample_selection_model_path: Optional[str] = None,
    weight_col: str = "day_gap",
    valid_days: int = 7,
    device: str = "cpu",
    seed: int = 2018,
) -> Dict[str, Any]:
    """
    直接迁移学习方法（带样本选择）
    
    从源域中选择与目标域相似的样本进行迁移学习
    
    参数:
        source_data: 源域训练数据 (AIR)
        target_data: 目标域训练数据 (HPG)
        test_data: 测试数据
        x_cols: 特征列名列表
        y_cols: 目标列名列表
        source_model_path: 源模型路径
        sample_selection_model_path: 保存模型路径
        weight_col: 权重列名
        valid_days: 验证天数
        device: 计算设备 ("cpu" 或 "gpu")
        seed: 随机种子
        
    返回:
        包含模型、预测结果、实际值和RMSLE分数的字典
    """
    model_base = ModelBase(device=device)
    
    # 准备数据集
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=target_data,
        test_data=test_data,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        valid_days=valid_days,
    )
    
    # 合并源域和目标域训练数据
    source_x, source_y = source_data[x_cols], source_data[y_cols].values.ravel()
    target_x, target_y = target_data[x_cols], target_data[y_cols].values.ravel()
    
    # 合并源域和目标域训练数据
    combined_x = pd.concat([source_x, target_x], ignore_index=True)
    combined_y = np.concatenate([source_y, target_y])
    
    # 创建权重 (目标域数据权重更高)
    source_weight = np.full(len(source_data), 0.5)  # 源域数据较低权重
    target_weight = np.ones(len(target_data))       # 目标域数据完整权重
    
    # 归一化权重
    combined_weight = np.concatenate([source_weight, target_weight])
    combined_weight = combined_weight / combined_weight.mean()
    
    # 创建训练数据集
    train_matrix = lgb.Dataset(combined_x, combined_y, weight=combined_weight, free_raw_data=False)
    
    # 训练模型
    model = model_base.train_or_load_lgbm(
        train_matrix,
        valid_matrix,
        sample_selection_model_path,
        num_threads=4,
        seed=seed,
    )
    
    # 评估
    predictions = model.predict(te_x, num_iteration=model.best_iteration)
    rmsle_score = np.sqrt(mean_squared_error(te_y, predictions))
    
    return {
        "model": model,
        "predictions": predictions,
        "actual": te_y,
        "rmsle": rmsle_score,
    }