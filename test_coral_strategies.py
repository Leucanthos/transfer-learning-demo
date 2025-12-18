#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同CORAL策略效果的脚本

根据用户需求，我们将测试几种不同的CORAL策略：
1. 标准CORAL（对所有特征进行对齐）
2. 自适应CORAL（只对相关性高的特征进行对齐）
3. 无CORAL（仅使用微调）
4. Partial CORAL（只对前50%相关性的特征进行对齐）
5. Threshold-based CORAL（设定不同阈值进行特征选择）

并将结果保存到日志文件中
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from src.model.base import ModelBase
from src.model.transfer import coral_align_source_to_target, finetune_from_source, _covariance, _whiten_and_color
from src.model.super_ensemble import adaptive_feature_alignment

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coral_strategies_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    """加载数据"""
    logger.info("Loading data...")
    
    # 加载AIR和HPG数据
    air_train = pd.read_parquet("./data/outputs/air_train.parquet")
    # air_test = pd.read_parquet("./data/outputs/air_test.parquet") 
    hpg_train = pd.read_parquet("./data/outputs/hpg_train.parquet")
    hpg_test = pd.read_parquet("./data/outputs/hpg_test.parquet")
    
    # 特征列 (排除非数值列)
    exclude_cols = ['air_store_id', 'visit_date', 'visitors']
    x_cols = [col for col in hpg_train.columns if col not in exclude_cols]
    y_cols = ['visitors']
    
    logger.info(f"Data loaded. AIR train: {len(air_train)}, HPG train: {len(hpg_train)}")
    logger.info(f"Feature columns count: {len(x_cols)}")
    
    return air_train, hpg_train, hpg_test, x_cols, y_cols

def prepare_data_for_coral(df, feature_cols):
    """为CORAL准备数据，移除非数值列"""
    # 只保留数值列
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    non_numeric_cols = set(feature_cols) - set(numeric_df.columns)
    if non_numeric_cols:
        logger.warning(f"Removed non-numeric columns: {non_numeric_cols}")
    
    # 合并回原数据框
    result_df = df.copy()
    result_df = result_df.drop(columns=list(non_numeric_cols))
    return result_df, numeric_df.columns.tolist()

def partial_coral_strategy(air_train, hpg_train, hpg_test, x_cols, y_cols, top_ratio=0.5):
    """
    Partial CORAL策略: 只对最相关的部分特征进行对齐
    """
    logger.info(f"Testing Partial CORAL strategy (top {top_ratio*100}% features)...")
    
    # 为CORAL准备数据，只保留数值特征
    air_train_clean, numeric_x_cols = prepare_data_for_coral(air_train, x_cols)
    hpg_train_clean, _ = prepare_data_for_coral(hpg_train, x_cols)
    
    # 计算每个特征在源域和目标域之间的相关性差异
    feature_similarities = []
    for col in numeric_x_cols:
        src_corr = air_train_clean[col].corr(air_train_clean['visitors']) if 'visitors' in air_train_clean.columns else 0
        tgt_corr = hpg_train_clean[col].corr(hpg_train_clean['visitors']) if 'visitors' in hpg_train_clean.columns else 0
        similarity = 1 - abs(src_corr - tgt_corr)
        feature_similarities.append(similarity)
    
    # 选择最相似的特征进行对齐
    num_top_features = int(len(numeric_x_cols) * top_ratio)
    top_indices = np.argsort(feature_similarities)[-num_top_features:]
    high_similarity_features = [numeric_x_cols[i] for i in top_indices]
    
    logger.info(f"Selecting top {len(high_similarity_features)} features for alignment")
    
    # 提取高相似度特征
    src_feat = air_train_clean[high_similarity_features].to_numpy(dtype=float)
    tgt_feat = hpg_train_clean[high_similarity_features].to_numpy(dtype=float)

    # 中心化
    src_mean = src_feat.mean(axis=0, keepdims=True)
    tgt_mean = tgt_feat.mean(axis=0, keepdims=True)
    src_centered = src_feat - src_mean

    # 协方差矩阵
    eps = 1e-6
    c_s = _covariance(src_centered, eps)
    c_t = _covariance(tgt_feat - tgt_mean, eps)

    # CORAL变换
    src_aligned = _whiten_and_color(src_centered, c_s, c_t) + tgt_mean

    # 写回数据（只更新高相似度特征）
    aligned_air_train = air_train_clean.copy()
    aligned_air_train[high_similarity_features] = src_aligned
    
    # 合并回原始数据的非特征列
    aligned_air_train_full = air_train.copy()
    aligned_air_train_full[numeric_x_cols] = aligned_air_train[numeric_x_cols]
    
    # 准备数据集
    model_base = ModelBase(device="gpu")
    aligned_train_matrix, aligned_valid_matrix, aligned_te_x, aligned_te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=aligned_air_train_full,
        test_data=hpg_test,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col='day_gap',
        valid_days=7,
    )
    
    # 训练模型
    aligned_model = model_base.train_or_load_lgbm(
        train_matrix=aligned_train_matrix,
        valid_matrix=aligned_valid_matrix,
        model_path=None,
        learning_rate=0.01,
        num_round=1000,
        num_threads=4,
        seed=2018,
        early_stopping_rounds=100,
    )
    
    # 评估模型
    results = model_base.evaluate_model(aligned_model, aligned_te_x, aligned_te_y)
    logger.info(f"Partial CORAL ({top_ratio*100}%) RMSLE: {results['rmsle']:.4f}")
    
    return results['rmsle']

def threshold_coral_strategy(air_train, hpg_train, hpg_test, x_cols, y_cols, threshold=0.5):
    """
    Threshold-based CORAL策略: 设定阈值选择特征
    """
    logger.info(f"Testing Threshold-based CORAL strategy (threshold={threshold})...")
    
    # 为CORAL准备数据，只保留数值特征
    air_train_clean, numeric_x_cols = prepare_data_for_coral(air_train, x_cols)
    hpg_train_clean, _ = prepare_data_for_coral(hpg_train, x_cols)
    
    # 计算每个特征在源域和目标域之间的相关性差异
    feature_similarities = []
    for col in numeric_x_cols:
        src_corr = air_train_clean[col].corr(air_train_clean['visitors']) if 'visitors' in air_train_clean.columns else 0
        tgt_corr = hpg_train_clean[col].corr(hpg_train_clean['visitors']) if 'visitors' in hpg_train_clean.columns else 0
        similarity = 1 - abs(src_corr - tgt_corr)
        feature_similarities.append(similarity)
    
    # 选择高于阈值的特征进行对齐
    high_similarity_features = [numeric_x_cols[i] for i, sim in enumerate(feature_similarities) if sim > threshold]
    
    if len(high_similarity_features) == 0:
        logger.warning("No features above threshold, using all features")
        high_similarity_features = numeric_x_cols
    
    logger.info(f"Selecting {len(high_similarity_features)} features above threshold {threshold}")
    
    # 提取高相似度特征
    src_feat = air_train_clean[high_similarity_features].to_numpy(dtype=float)
    tgt_feat = hpg_train_clean[high_similarity_features].to_numpy(dtype=float)

    # 中心化
    src_mean = src_feat.mean(axis=0, keepdims=True)
    tgt_mean = tgt_feat.mean(axis=0, keepdims=True)
    src_centered = src_feat - src_mean

    # 协方差矩阵
    eps = 1e-6
    c_s = _covariance(src_centered, eps)
    c_t = _covariance(tgt_feat - tgt_mean, eps)

    # CORAL变换
    src_aligned = _whiten_and_color(src_centered, c_s, c_t) + tgt_mean

    # 写回数据（只更新高相似度特征）
    aligned_air_train = air_train_clean.copy()
    aligned_air_train[high_similarity_features] = src_aligned
    
    # 合并回原始数据的非特征列
    aligned_air_train_full = air_train.copy()
    aligned_air_train_full[numeric_x_cols] = aligned_air_train[numeric_x_cols]
    
    # 准备数据集
    model_base = ModelBase(device="gpu")
    aligned_train_matrix, aligned_valid_matrix, aligned_te_x, aligned_te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=aligned_air_train_full,
        test_data=hpg_test,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col='day_gap',
        valid_days=7,
    )
    
    # 训练模型
    aligned_model = model_base.train_or_load_lgbm(
        train_matrix=aligned_train_matrix,
        valid_matrix=aligned_valid_matrix,
        model_path=None,
        learning_rate=0.01,
        num_round=1000,
        num_threads=4,
        seed=2018,
        early_stopping_rounds=100,
    )
    
    # 评估模型
    results = model_base.evaluate_model(aligned_model, aligned_te_x, aligned_te_y)
    logger.info(f"Threshold-based CORAL (threshold={threshold}) RMSLE: {results['rmsle']:.4f}")
    
    return results['rmsle']

def standard_coral_strategy(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """
    标准CORAL策略: 对所有特征进行对齐
    """
    logger.info("Testing Standard CORAL strategy...")
    
    # 为CORAL准备数据，只保留数值特征
    air_train_clean, numeric_x_cols = prepare_data_for_coral(air_train, x_cols)
    hpg_train_clean, _ = prepare_data_for_coral(hpg_train, x_cols)
    hpg_test_clean, _ = prepare_data_for_coral(hpg_test, x_cols)
    
    # 使用标准CORAL对齐源域特征到目标域
    coral_air_train = coral_align_source_to_target(
        source_df=air_train_clean,
        target_df=hpg_train_clean,
        feature_cols=numeric_x_cols,
    )
    
    # 合并回原始数据的非特征列
    coral_air_train_full = air_train.copy()
    coral_air_train_full[numeric_x_cols] = coral_air_train[numeric_x_cols]
    
    # 准备数据集
    model_base = ModelBase(device="gpu")
    coral_train_matrix, coral_valid_matrix, coral_te_x, coral_te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=coral_air_train_full,
        test_data=hpg_test,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col='day_gap',
        valid_days=7,
    )
    
    # 训练模型
    coral_model = model_base.train_or_load_lgbm(
        train_matrix=coral_train_matrix,
        valid_matrix=coral_valid_matrix,
        model_path=None,
        learning_rate=0.01,
        num_round=1000,
        num_threads=4,
        seed=2018,
        early_stopping_rounds=100,
    )
    
    # 评估模型
    results = model_base.evaluate_model(coral_model, coral_te_x, coral_te_y)
    logger.info(f"Standard CORAL RMSLE: {results['rmsle']:.4f}")
    
    return results['rmsle']

def adaptive_coral_strategy(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """
    自适应CORAL策略: 只对相关性高的特征进行对齐（类似Super Ensemble中的方法）
    """
    logger.info("Testing Adaptive CORAL strategy...")
    
    # 为CORAL准备数据，只保留数值特征
    air_train_clean, numeric_x_cols = prepare_data_for_coral(air_train, x_cols)
    hpg_train_clean, _ = prepare_data_for_coral(hpg_train, x_cols)
    hpg_test_clean, _ = prepare_data_for_coral(hpg_test, x_cols)
    
    # 使用自适应特征对齐
    aligned_air_train = adaptive_feature_alignment(
        source_df=air_train_clean,
        target_df=hpg_train_clean,
        feature_cols=numeric_x_cols,
        similarity_threshold=0.7
    )
    
    # 合并回原始数据的非特征列
    aligned_air_train_full = air_train.copy()
    aligned_air_train_full[numeric_x_cols] = aligned_air_train[numeric_x_cols]
    
    # 准备数据集
    model_base = ModelBase(device="gpu")
    aligned_train_matrix, aligned_valid_matrix, aligned_te_x, aligned_te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=aligned_air_train_full,
        test_data=hpg_test,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col='day_gap',
        valid_days=7,
    )
    
    # 训练模型
    aligned_model = model_base.train_or_load_lgbm(
        train_matrix=aligned_train_matrix,
        valid_matrix=aligned_valid_matrix,
        model_path=None,
        learning_rate=0.01,
        num_round=1000,
        num_threads=4,
        seed=2018,
        early_stopping_rounds=100,
    )
    
    # 评估模型
    results = model_base.evaluate_model(aligned_model, aligned_te_x, aligned_te_y)
    logger.info(f"Adaptive CORAL RMSLE: {results['rmsle']:.4f}")
    
    return results['rmsle']

def no_coral_strategy(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """
    无CORAL策略: 仅使用微调
    """
    logger.info("Testing No CORAL (Fine-tuning only) strategy...")
    
    # 微调策略
    finetune_result = finetune_from_source(
        train_data=hpg_train,
        test_data=hpg_test,
        x_cols=x_cols,
        y_cols=y_cols,
        source_model_path="./lgbm_weights/air_model.pkl",
        finetuned_model_path=None,
        weight_col="day_gap",
        valid_days=7,
        learning_rate=0.001,
        num_round=1000,
        device="gpu",
    )
    
    logger.info(f"No CORAL (Fine-tuning) RMSLE: {finetune_result['rmsle']:.4f}")
    return finetune_result['rmsle']

def main():
    """主函数"""
    logger.info("="*50)
    logger.info("Starting CORAL Strategies Comparison Test")
    logger.info("="*50)
    
    # 加载数据
    air_train, hpg_train, hpg_test, x_cols, y_cols = load_data()
    
    # 存储结果
    results = {}
    
    # 测试各种策略
    try:
        results['no_coral'] = no_coral_strategy(air_train, hpg_train, hpg_test, x_cols, y_cols)
    except Exception as e:
        logger.error(f"Error in No CORAL strategy: {e}")
        results['no_coral'] = None
    
    try:
        results['standard_coral'] = standard_coral_strategy(air_train, hpg_train, hpg_test, x_cols, y_cols)
    except Exception as e:
        logger.error(f"Error in Standard CORAL strategy: {e}")
        results['standard_coral'] = None
    
    try:
        results['adaptive_coral'] = adaptive_coral_strategy(air_train, hpg_train, hpg_test, x_cols, y_cols)
    except Exception as e:
        logger.error(f"Error in Adaptive CORAL strategy: {e}")
        results['adaptive_coral'] = None
        
    try:
        results['partial_coral_30'] = partial_coral_strategy(air_train, hpg_train, hpg_test, x_cols, y_cols, 0.3)
    except Exception as e:
        logger.error(f"Error in Partial CORAL (30%) strategy: {e}")
        results['partial_coral_30'] = None
        
    try:
        results['partial_coral_50'] = partial_coral_strategy(air_train, hpg_train, hpg_test, x_cols, y_cols, 0.5)
    except Exception as e:
        logger.error(f"Error in Partial CORAL (50%) strategy: {e}")
        results['partial_coral_50'] = None
        
    try:
        results['threshold_coral_03'] = threshold_coral_strategy(air_train, hpg_train, hpg_test, x_cols, y_cols, 0.3)
    except Exception as e:
        logger.error(f"Error in Threshold CORAL (0.3) strategy: {e}")
        results['threshold_coral_03'] = None
        
    try:
        results['threshold_coral_05'] = threshold_coral_strategy(air_train, hpg_train, hpg_test, x_cols, y_cols, 0.5)
    except Exception as e:
        logger.error(f"Error in Threshold CORAL (0.5) strategy: {e}")
        results['threshold_coral_05'] = None
    
    # 输出最终结果
    logger.info("="*50)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*50)
    
    for strategy, rmsle in results.items():
        if rmsle is not None:
            logger.info(f"{strategy}: {rmsle:.4f}")
        else:
            logger.info(f"{strategy}: FAILED")
    
    logger.info("="*50)
    logger.info("Test completed. Results saved to coral_strategies_test.log")

if __name__ == "__main__":
    main()