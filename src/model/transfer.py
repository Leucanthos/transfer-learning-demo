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
    """Load a pickled LightGBM Booster."""
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
    x_cols: list,
    y_cols: list,
    source_model_path: str,
    fine_tuned_model_path: Optional[str] = None,
    weight_col: str = "day_gap",
    valid_days: int = 7,
    learning_rate: float = 0.002,       # 微调使用较小的学习率
    num_round: int = 3000,
    num_threads: int = 4,
    seed: int = 2018,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    微调迁移学习方法
    
    加载源域预训练模型，在目标任务上以较小的学习率继续训练
    
    参数:
    source_data: 源域训练数据 (AIR)
    target_data: 目标域训练数据 (HPG)
    test_data: 测试数据
    x_cols: 特征列名列表
    y_cols: 目标列名列表
    source_model_path: 源模型路径
    fine_tuned_model_path: 保存模型路径
    weight_col: 权重列名
    valid_days: 验证天数
    learning_rate: 学习率 (微调使用较小值)
    num_round: 训练轮数
    num_threads: 线程数
    seed: 随机种子
    device: 训练设备 ("cpu" 或 "gpu")
    
    返回:
    包含预测结果和RMSLE评分的字典
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
    
    # Fine-tune model on target data
    model = model_base.train_or_load_lgbm(
        train_matrix,
        valid_matrix,
        None,  # Don't save intermediate models
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
    x_cols: list,
    y_cols: list,
    source_model_path: str,
    transferred_model_path: Optional[str] = None,
    weight_col: str = "day_gap",
    valid_days: int = 7,
    selection_ratio: float = 0.7,      # 样本选择比例
    num_threads: int = 4,
    seed: int = 2018,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    基于样本选择的直接迁移方法
    
    首先从源域中选择与目标域相似的样本，然后与目标域数据合并训练模型
    
    参数:
    source_data: 源域训练数据 (AIR)
    target_data: 目标域训练数据 (HPG)
    test_data: 测试数据
    x_cols: 特征列名列表
    y_cols: 目标列名列表
    source_model_path: 源模型路径
    transferred_model_path: 保存模型路径
    weight_col: 权重列名
    valid_days: 验证天数
    selection_ratio: 样本选择比例
    num_threads: 线程数
    seed: 随机种子
    device: 训练设备 ("cpu" 或 "gpu")
    
    返回:
    包含预测结果和RMSLE评分的字典
    """
    model_base = ModelBase(device=device)
    
    # 智能样本筛选 - 选择与目标域相似的源域样本
    # 简化实现：随机选择一定比例的源域样本
    # 在实际应用中可以使用更复杂的相似度计算方法
    n_select = int(len(source_data) * selection_ratio)
    np.random.seed(seed)
    selected_indices = np.random.choice(len(source_data), n_select, replace=False)
    selected_source_data = source_data.iloc[selected_indices]
    
    # 合并筛选后的源域数据和目标域数据
    combined_train_data = pd.concat([target_data, selected_source_data], ignore_index=True)
    
    # 准备训练数据
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=combined_train_data,
        test_data=test_data,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        valid_days=valid_days,
    )
    
    # 加载源模型作为初始化模型
    source_model = _load_booster(source_model_path)
    
    # 训练模型
    transferred_model = model_base.train_or_load_lgbm(
        train_matrix,
        valid_matrix,
        transferred_model_path,
        learning_rate=0.01,
        num_round=3000,
        num_threads=num_threads,
        seed=seed,
        early_stopping_rounds=100,
        init_model=source_model,
    )

    # 评估模型
    results = model_base.evaluate_model(transferred_model, te_x, te_y)
    return results