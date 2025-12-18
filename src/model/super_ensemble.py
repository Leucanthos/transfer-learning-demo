import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans

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


def smart_sample_selection(source_data: pd.DataFrame, 
                          target_data: pd.DataFrame, 
                          feature_cols: list, 
                          n_clusters: int = 20,
                          selection_ratio: float = 0.7) -> pd.DataFrame:
    """
    智能样本筛选方法，使用聚类方法识别源域中与目标域相似的样本
    
    参数:
    source_data: 源域数据
    target_data: 目标域数据
    feature_cols: 特征列名列表
    n_clusters: 聚类数量
    selection_ratio: 选择样本的比例
    
    返回:
    筛选后的源域数据
    """
    # 使用KMeans聚类将目标域数据分成n_clusters个簇
    target_features = target_data[feature_cols].values
    kmeans_target = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    target_clusters = kmeans_target.fit_predict(target_features)
    
    # 计算每个簇的中心
    target_cluster_centers = kmeans_target.cluster_centers_
    
    # 对源域数据进行聚类
    source_features = source_data[feature_cols].values
    source_clusters = kmeans_target.predict(source_features)
    
    # 计算源域每个样本到目标域簇中心的距离
    source_distances = np.zeros(len(source_features))
    for i in range(len(source_features)):
        cluster_id = source_clusters[i]
        source_distances[i] = np.linalg.norm(source_features[i] - target_cluster_centers[cluster_id])
    
    # 选择距离最近的一部分样本
    threshold_index = int(len(source_distances) * selection_ratio)
    closest_indices = np.argpartition(source_distances, threshold_index)[:threshold_index]
    useful_samples = source_data.iloc[closest_indices]
    
    return useful_samples


def partial_feature_alignment(source_df: pd.DataFrame, 
                           target_df: pd.DataFrame, 
                           feature_cols: list, 
                           top_ratio: float = 0.1) -> pd.DataFrame:
    """
    Partial CORAL方法，选择最相关的特征进行对齐（Top特征）
    
    参数:
    source_df: 源域数据
    target_df: 目标域数据
    feature_cols: 特征列名列表
    top_ratio: 选择特征的比例
    
    返回:
    对齐后的源域数据
    """
    src = source_df.copy()
    tgt = target_df.copy()

    # 计算每个特征在源域和目标域之间的相关性差异
    feature_similarities = []
    for col in feature_cols:
        src_corr = src[col].corr(src['visitors']) if 'visitors' in src.columns else 0
        tgt_corr = tgt[col].corr(tgt['visitors']) if 'visitors' in tgt.columns else 0
        similarity = 1 - abs(src_corr - tgt_corr)
        feature_similarities.append(similarity)
    
    # 选择最相似的特征进行对齐
    num_top_features = int(len(feature_cols) * top_ratio)
    top_indices = np.argsort(feature_similarities)[-num_top_features:]
    high_similarity_features = [feature_cols[i] for i in top_indices]
    
    logger.info(f"Selected top {len(high_similarity_features)} features for alignment")
    
    # 提取高相似度特征
    src_feat = src[high_similarity_features].to_numpy(dtype=float)
    tgt_feat = tgt[high_similarity_features].to_numpy(dtype=float)

    # 中心化
    src_mean = src_feat.mean(axis=0, keepdims=True)
    tgt_mean = tgt_feat.mean(axis=0, keepdims=True)
    src_centered = src_feat - src_mean

    # 协方差矩阵
    eps = 1e-6
    c_s = np.cov(src_centered, rowvar=False) + np.eye(len(high_similarity_features)) * eps
    c_t = np.cov(tgt_feat - tgt_mean, rowvar=False) + np.eye(len(high_similarity_features)) * eps

    # 特征对齐变换
    d_s, e_s = np.linalg.eigh(c_s)
    d_t, e_t = np.linalg.eigh(c_t)

    # 构建白化和着色矩阵
    w_s = e_s @ np.diag(1.0 / np.sqrt(np.maximum(d_s, 1e-12))) @ e_s.T
    c_t_half = e_t @ np.diag(np.sqrt(np.maximum(d_t, 1e-12))) @ e_t.T

    src_aligned = (src_centered @ w_s) @ c_t_half + tgt_mean

    # 写回数据（只更新高相似度特征）
    src[high_similarity_features] = src_aligned
    return src


def evaluate_stage_loss(model_base, train_data, test_data, x_cols, y_cols, weight_col, valid_days, stage_name, init_model=None):
    """
    评估特定阶段的损失
    
    参数:
    model_base: 模型基类实例
    train_data: 训练数据
    test_data: 测试数据
    x_cols: 特征列
    y_cols: 标签列
    weight_col: 权重列
    valid_days: 验证天数
    stage_name: 阶段名称
    init_model: 初始化模型
    
    返回:
    该阶段的RMSLE分数
    """
    logger.info(f"Evaluating loss at {stage_name}...")
    
    # 准备数据
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=train_data,
        test_data=test_data,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        valid_days=valid_days,
    )
    
    # 训练模型
    model = model_base.train_or_load_lgbm(
        train_matrix,
        valid_matrix,
        None,  # 不保存中间模型
        learning_rate=0.01,
        num_round=500,
        num_threads=4,
        seed=2018,
        early_stopping_rounds=100,
        init_model=init_model,
    )
    
    # 评估模型
    results = model_base.evaluate_model(model, te_x, te_y)
    logger.info(f"{stage_name} RMSLE: {results['rmsle']:.4f}")
    
    return results['rmsle']


def super_ensemble_transfer_learning(
    source_data: pd.DataFrame,
    target_data: pd.DataFrame,
    test_data: pd.DataFrame,
    x_cols: list,
    y_cols: list,
    source_model_path: str,
    super_ensemble_model_path: Optional[str] = None,
    weight_col: str = "day_gap",
    valid_days: int = 7,
    learning_rate_phase1: float = 0.01,     # 第一阶段学习率
    learning_rate_phase2: float = 0.002,    # 第二阶段学习率
    num_round_phase1: int = 500,            # 第一阶段轮数
    num_round_phase2: int = 3000,           # 第二阶段轮数
    num_threads: int = 4,
    seed: int = 2018,
    device: str = "cpu",
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