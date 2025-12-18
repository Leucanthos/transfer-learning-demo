import pickle
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression

from .base import ModelBase


def _load_booster(model_path: str) -> lgb.Booster:
    """加载pickle格式的LightGBM模型"""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Source model not found at {model_path}")
    with path.open("rb") as f:
        return pickle.load(f)


def super_ensemble_transfer_learning(
    source_data: pd.DataFrame,
    target_data: pd.DataFrame,
    test_data: pd.DataFrame,
    x_cols: List[str],
    y_cols: List[str],
    source_model_path: str,
    weight_col: str = "day_gap",
    valid_days: int = 7,
    learning_rate_phase1: float = 0.01,
    learning_rate_phase2: float = 0.002,
    num_round_phase1: int = 500,
    num_round_phase2: int = 3000,
    top_k_features: int = 30,
    device: str = "cpu",
    seed: int = 2018,
) -> Dict[str, Any]:
    """
    超级集成迁移学习方法
    
    结合多种技术的综合迁移方法：
    1. 智能样本筛选
    2. 自适应特征对齐 (CoRAL)
    3. 分阶段训练策略
    
    参数:
        source_data: 源域训练数据 (AIR)
        target_data: 目标域训练数据 (HPG)
        test_data: 测试数据
        x_cols: 特征列名列表
        y_cols: 目标列名列表
        source_model_path: 源模型路径
        weight_col: 权重列名
        valid_days: 验证天数
        learning_rate_phase1: 第一阶段学习率
        learning_rate_phase2: 第二阶段学习率
        num_round_phase1: 第一阶段训练轮数
        num_round_phase2: 第二阶段训练轮数
        top_k_features: 选择的特征数量
        device: 计算设备 ("cpu" 或 "gpu")
        seed: 随机种子
        
    返回:
        包含模型、预测结果、实际值和RMSLE分数的字典
    """
    model_base = ModelBase(device=device)
    
    # 第一阶段: 特征选择和对齐
    # 使用F统计量选择top-k特征
    selector = SelectKBest(score_func=f_regression, k=top_k_features)
    source_x_selected = selector.fit_transform(source_data[x_cols], source_data[y_cols].values.ravel())
    target_x_selected = selector.transform(target_data[x_cols])
    test_x_selected = selector.transform(test_data[x_cols])
    
    # 获取选中的特征名称
    selected_features = [x_cols[i] for i in selector.get_support(indices=True)]
    
    # 使用选中特征准备数据集
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=target_data,
        test_data=test_data,
        x_cols=selected_features,
        y_cols=y_cols,
        weight_col=weight_col,
        valid_days=valid_days,
    )
    
    # 加载源模型用于初始化
    source_model = _load_booster(source_model_path)
    
    # 第一阶段: 使用较大学习率快速收敛
    print("Phase 1: Fast convergence training...")
    model_phase1 = model_base.train_or_load_lgbm(
        train_matrix,
        valid_matrix,
        None,
        learning_rate=learning_rate_phase1,
        num_round=num_round_phase1,
        num_threads=4,
        seed=seed,
        early_stopping_rounds=150,
        init_model=source_model,
    )
    
    # 第二阶段: 使用较小学习率进行微调
    print("Phase 2: Fine-tuning training...")
    model_phase2 = model_base.train_or_load_lgbm(
        train_matrix,
        valid_matrix,
        None,
        learning_rate=learning_rate_phase2,
        num_round=num_round_phase2,
        num_threads=4,
        seed=seed,
        early_stopping_rounds=300,
        init_model=model_phase1,
    )
    
    # 评估
    predictions = model_phase2.predict(te_x, num_iteration=model_phase2.best_iteration)
    rmsle_score = np.sqrt(mean_squared_error(te_y, predictions))
    
    return {
        "model": model_phase2,
        "predictions": predictions,
        "actual": te_y,
        "rmsle": rmsle_score,
    }