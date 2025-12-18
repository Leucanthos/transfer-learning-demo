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
    source_model = _load_booster(source_model_path)
    
    # 为目标域数据生成伪标签
    target_features = target_data[x_cols]
    pseudo_labels = source_model.predict(target_features)
    
    # 创建带伪标签的目标域数据
    pseudo_target_data = target_data.copy()
    pseudo_target_data['visitors'] = pseudo_labels
    
    # 合并源域数据和带伪标签的目标域数据
    combined_train_data = pd.concat([source_data, pseudo_target_data], ignore_index=True)
    
    # 准备训练数据
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=combined_train_data,
        test_data=test_data,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        valid_days=valid_days,
    )
    
    # 训练模型
    pseudo_model = model_base.train_or_load_lgbm(
        train_matrix,
        valid_matrix,
        pseudo_model_path,
        learning_rate=0.01,
        num_round=3000,
        num_threads=4,
        seed=seed,
        early_stopping_rounds=100,
    )

    # 评估模型
    results = model_base.evaluate_model(pseudo_model, te_x, te_y)
    return results


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
    
    # 准备目标域数据
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=target_data,
        test_data=test_data,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        valid_days=valid_days,
    )
    
    # 加载源模型作为初始化模型
    source_model = _load_booster(source_model_path)
    
    # 在目标域上微调模型
    fine_tuned_model = model_base.train_or_load_lgbm(
        train_matrix,
        valid_matrix,
        fine_tuned_model_path,
        learning_rate=learning_rate,
        num_round=num_round,
        num_threads=num_threads,
        seed=seed,
        early_stopping_rounds=100,
        init_model=source_model,  # 使用源模型作为初始化
    )

    # 评估模型
    results = model_base.evaluate_model(fine_tuned_model, te_x, te_y)
    return results


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