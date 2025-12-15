import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .base import ModelBase


def _load_booster(model_path: str) -> lgb.Booster:
    """Load a pickled LightGBM Booster."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Source model not found at {model_path}")
    with path.open("rb") as f:
        return pickle.load(f)


def weighted_domain_adaptation(
    source_data,
    target_data,
    test_data,
    x_cols,
    y_cols,
    source_model_path: str,
    adapted_model_path: Optional[str] = None,
    weight_col: str = "day_gap",
    valid_days: int = 7,
    source_weight: float = 0.3,
    target_weight: float = 0.7,
    learning_rate: float = 0.005,
    num_round: int = 2000,
    num_threads: int = 4,
    seed: int = 2018,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    使用加权域适应方法进行迁移学习，结合源域和目标域数据
    
    该方法通过给源域和目标域数据分配不同权重来训练模型，而不是直接微调
    """
    model_base = ModelBase(device=device)
    
    # 准备目标域数据（用于验证和测试）
    _, target_valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=target_data,
        test_data=test_data,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        valid_days=valid_days,
    )

    # 加载源模型
    source_model = _load_booster(source_model_path)

    # 训练新模型，使用源域数据作为增强训练集
    print(f"Training with weighted domain adaptation (source:{source_weight}, target:{target_weight}) on {device.upper()}...")
    
    # 获取源域和目标域的数据
    target_train_data = target_data[x_cols].values
    target_train_labels = target_data[y_cols].values.ravel()
    
    source_train_data = source_data[x_cols].values
    source_train_labels = source_data[y_cols].values.ravel()
    
    # 计算权重
    target_weight_raw = target_data[weight_col].values.ravel()
    source_weight_raw = source_data[weight_col].values.ravel()
    
    # 计算权重：day_gap越小权重越大
    epsilon = 1e-8
    target_weights = np.where(target_weight_raw >= 0, 
                              1 / (1 + target_weight_raw + epsilon), 
                              1 / (1 - target_weight_raw + epsilon))
    source_weights = np.where(source_weight_raw >= 0, 
                              1 / (1 + source_weight_raw + epsilon), 
                              1 / (1 - source_weight_raw + epsilon))
    
    # 归一化权重
    target_weights = target_weights / np.mean(target_weights)
    source_weights = source_weights / np.mean(source_weights)
    
    # 应用域权重
    target_weights = target_weights * target_weight
    source_weights = source_weights * source_weight
    
    # 合并数据集
    combined_train_data = np.vstack([target_train_data, source_train_data])
    combined_train_labels = np.concatenate([target_train_labels, source_train_labels])
    combined_train_weights = np.concatenate([target_weights, source_weights])
    
    # 创建合并后的数据集
    combined_train_matrix = lgb.Dataset(combined_train_data, 
                                       label=combined_train_labels, 
                                       weight=combined_train_weights)
    
    # 训练模型
    adapted_model = model_base.train_or_load_lgbm(
        combined_train_matrix,
        target_valid_matrix,
        adapted_model_path,
        learning_rate=learning_rate,
        num_round=num_round,
        num_threads=num_threads,
        seed=seed,
        early_stopping_rounds=300,
        init_model=source_model,
    )

    return model_base.evaluate_model(adapted_model, te_x, te_y)


def pseudo_labeling_transfer(
    source_data,
    target_unlabeled_data,
    test_data,
    x_cols,
    y_cols,
    source_model_path: str,
    pseudo_model_path: Optional[str] = None,
    weight_col: str = "day_gap",
    valid_days: int = 7,
    learning_rate: float = 0.005,
    num_round: int = 2000,
    num_threads: int = 4,
    seed: int = 2018,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    使用伪标签进行迁移学习
    
    该方法使用源域训练的模型为目标域无标签数据生成伪标签，
    然后将这些伪标签数据与目标域的真实数据一起训练模型
    """
    model_base = ModelBase(device=device)
    
    # 加载源模型
    source_model = _load_booster(source_model_path)
    
    # 为目标域无标签数据生成伪标签
    unlabeled_features = target_unlabeled_data[x_cols]
    pseudo_labels = source_model.predict(unlabeled_features)
    
    # 创建伪标签数据框
    pseudo_labeled_data = target_unlabeled_data.copy()
    pseudo_labeled_data[y_cols[0]] = pseudo_labels
    
    # 结合目标域的真实数据和伪标签数据
    combined_train_data = pd.concat([target_unlabeled_data, pseudo_labeled_data], axis=0)
    
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
    print(f"Training with pseudo-labeling transfer on {device.upper()}...")
    pseudo_model = model_base.train_or_load_lgbm(
        train_matrix,
        valid_matrix,
        pseudo_model_path,
        learning_rate=learning_rate,
        num_round=num_round,
        num_threads=num_threads,
        seed=seed,
        early_stopping_rounds=300,
        init_model=source_model,
    )

    if pseudo_model_path:
        path = Path(pseudo_model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(pseudo_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Pseudo labeling model saved to {pseudo_model_path}")

    return evaluate_model(pseudo_model, te_x, te_y)


def adversarial_domain_adaptation(
    source_data,
    target_data,
    test_data,
    x_cols,
    y_cols,
    source_model_path: str,
    adv_model_path: Optional[str] = None,
    weight_col: str = "day_gap",
    valid_days: int = 7,
    learning_rate: float = 0.003,
    num_round: int = 2000,
    num_threads: int = 4,
    seed: int = 2018,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    尝试对抗域适应方法
    
    该方法通过特征级的对抗训练减少源域和目标域之间的差异
    """
    model_base = ModelBase(device=device)
    
    # 准备数据
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=target_data,
        test_data=test_data,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        valid_days=valid_days,
    )
    
    # 加载源模型
    source_model = _load_booster(source_model_path)
    
    # 提取源域和目标域的特征表示
    source_features = source_data[x_cols].values
    target_features = target_data[x_cols].values
    
    # 计算特征重要性差异并调整目标域数据
    # 这是一种简化的对抗方法，通过特征对齐来减少域间差异
    
    # 获取源模型的特征重要性
    source_importance = source_model.feature_importance()
    
    # 标准化特征重要性
    source_importance_norm = source_importance / np.sum(source_importance)
    
    # 调整目标域数据权重，使得重要特征更加突出
    adjusted_target_data = target_data.copy()
    
    # 基于源模型的特征重要性调整目标域特征
    for i, col in enumerate(x_cols):
        # 增强重要特征的影响
        adjusted_target_data[col] = target_data[col] * (1 + source_importance_norm[i])
    
    # 使用调整后的数据重新训练
    adjusted_train_matrix, adjusted_valid_matrix, _, _ = model_base.prepare_lgbm_dataset_with_weights(
        train_data=adjusted_target_data,
        test_data=test_data,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        valid_days=valid_days,
    )
    
    # 训练模型
    print(f"Training with adversarial domain adaptation on {device.upper()}...")
    adv_model = model_base.train_or_load_lgbm(
        adjusted_train_matrix,
        adjusted_valid_matrix,
        adv_model_path,
        learning_rate=learning_rate,
        num_round=num_round,
        num_threads=num_threads,
        seed=seed,
        early_stopping_rounds=300,
        init_model=source_model,
    )

    if adv_model_path:
        path = Path(adv_model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(adv_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Adversarial model saved to {adv_model_path}")

    return evaluate_model(adv_model, te_x, te_y)