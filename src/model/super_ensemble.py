import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression

from .base import ModelBase


def _load_booster(model_path: str) -> lgb.Booster:
    """Load a pickled LightGBM Booster."""
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
    超级集成迁移学习方法（单模型方法）
    
    这是一个单一模型的迁移学习方法，结合了多种技术：
    1. 智能样本筛选 - 选择源域中与目标域最相关的样本
    2. Partial特征对齐 - 选择最相关的特征进行对齐
    3. 两阶段训练策略 - 先快速收敛，再精细调优
    
    参数:
    source_data: 源域训练数据 (AIR)
    target_data: 目标域训练数据 (HPG)
    test_data: 测试数据
    x_cols: 特征列名列表
    y_cols: 目标列名列表
    source_model_path: 源模型路径
    super_ensemble_model_path: 保存模型路径
    weight_col: 权重列名
    valid_days: 验证天数
    learning_rate_phase1: 第一阶段学习率
    learning_rate_phase2: 第二阶段学习率
    num_round_phase1: 第一阶段训练轮数
    num_round_phase2: 第二阶段训练轮数
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
    
    # 加载源模型
    source_model = _load_booster(source_model_path)
    
    # 评估初始模型在目标域上的性能
    initial_rmsle = evaluate_stage_loss(model_base, target_data, test_data, x_cols, y_cols, weight_col, valid_days, "Initial Target-only Model")
    
    # 第一阶段：智能样本筛选
    logger.info("Stage 1: Smart sample selection...")
    selected_source_data = smart_sample_selection(source_data, target_data, x_cols, n_clusters=20, selection_ratio=0.7)
    
    # 评估智能样本筛选后对目标域的影响
    selected_train_data = pd.concat([target_data, selected_source_data], ignore_index=True)
    sample_selection_rmsle = evaluate_stage_loss(model_base, selected_train_data, test_data, x_cols, y_cols, weight_col, valid_days, "After Sample Selection", source_model)
    
    # 第二阶段：Partial特征对齐
    logger.info("Stage 2: Partial feature alignment...")
    aligned_source_data = partial_feature_alignment(selected_source_data, target_data, x_cols, top_ratio=0.1)
    
    # 评估特征对齐后对目标域的影响
    aligned_train_data = pd.concat([target_data, aligned_source_data], ignore_index=True)
    feature_alignment_rmsle = evaluate_stage_loss(model_base, aligned_train_data, test_data, x_cols, y_cols, weight_col, valid_days, "After Feature Alignment", source_model)
    
    # 第三阶段：数据增强
    logger.info("Stage 3: Data augmentation...")
    enhanced_train_data = pd.concat([target_data, aligned_source_data], ignore_index=True)
    
    # 评估数据增强后对目标域的影响
    data_augmentation_rmsle = evaluate_stage_loss(model_base, enhanced_train_data, test_data, x_cols, y_cols, weight_col, valid_days, "After Data Augmentation", source_model)
    
    # 第四阶段：两阶段训练
    logger.info(f"Stage 4: Two-phase training on {device.upper()}...")
    
    # 重新准备增强后的训练数据
    enhanced_train_matrix, enhanced_valid_matrix, _, _ = model_base.prepare_lgbm_dataset_with_weights(
        train_data=enhanced_train_data,
        test_data=test_data,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        valid_days=valid_days,
    )
    
    # 第一阶段：使用较高的学习率快速收敛
    logger.info(f"Phase 1: Fast convergence (lr={learning_rate_phase1}, rounds={num_round_phase1})")
    fast_model = model_base.train_or_load_lgbm(
        enhanced_train_matrix,
        enhanced_valid_matrix,
        None,  # 不保存中间模型
        learning_rate=learning_rate_phase1,
        num_round=num_round_phase1,
        num_threads=num_threads,
        seed=seed,
        early_stopping_rounds=100,
        init_model=source_model,
    )
    
    # 评估第一阶段训练后对目标域的影响
    phase1_rmsle = model_base.evaluate_model(fast_model, te_x, te_y)['rmsle']
    logger.info(f"After Phase 1 Training RMSLE: {phase1_rmsle:.4f}")
    
    # 第二阶段：使用较低的学习率精细调优
    logger.info(f"Phase 2: Fine-tuning (lr={learning_rate_phase2}, rounds={num_round_phase2})")
    super_ensemble_model = model_base.train_or_load_lgbm(
        enhanced_train_matrix,
        enhanced_valid_matrix,
        super_ensemble_model_path,
        learning_rate=learning_rate_phase2,
        num_round=num_round_phase2,
        num_threads=num_threads,
        seed=seed,
        early_stopping_rounds=500,
        init_model=fast_model,
    )

    final_results = model_base.evaluate_model(super_ensemble_model, te_x, te_y)
    
    # 记录各阶段的提升效果
    logger.info("=" * 50)
    logger.info("STEP-BY-STEP IMPROVEMENT SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Initial Target-only Model:                 {initial_rmsle:.4f}")
    logger.info(f"After Sample Selection:                   {sample_selection_rmsle:.4f} ({initial_rmsle - sample_selection_rmsle:+.4f})")
    logger.info(f"After Feature Alignment:                 {feature_alignment_rmsle:.4f} ({sample_selection_rmsle - feature_alignment_rmsle:+.4f})")
    logger.info(f"After Data Augmentation:                 {data_augmentation_rmsle:.4f} ({feature_alignment_rmsle - data_augmentation_rmsle:+.4f})")
    logger.info(f"After Phase 1 Training:                  {phase1_rmsle:.4f} ({data_augmentation_rmsle - phase1_rmsle:+.4f})")
    logger.info(f"Final Super Ensemble Model:               {final_results['rmsle']:.4f} ({phase1_rmsle - final_results['rmsle']:+.4f})")
    logger.info("=" * 50)
    logger.info("Total Improvement:                        {:.4f}".format(initial_rmsle - final_results['rmsle']))
    logger.info("=" * 50)
    
    return final_results