#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的迁移学习实验运行脚本

该脚本整合了所有迁移学习方法的实验，包括：
1. 基线方法
2. 伪标签迁移学习
3. 微调迁移学习
4. 对抗域适应
5. 超级集成方法

支持的功能：
- 运行单一方法实验
- 运行所有方法实验
- 参数调优实验
- 结果可视化
"""

import os
import sys
import argparse
import logging
import warnings
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import date, timedelta
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transfer_learning.transfer import pseudo_label_transfer_learning, fine_tune_transfer_learning, direct_transfer_with_sample_selection
from transfer_learning.advanced_transfer import adversarial_domain_adaptation
from transfer_learning.super_ensemble import super_ensemble_transfer_learning
from transfer_learning.ensemble_transfer import true_ensemble_transfer_learning
from transfer_learning.baseline import train_or_load_lgbm, evaluate_model
from transfer_learning.base import ModelBase

# 设置随机种子
np.random.seed(2018)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiments.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# 设置matplotlib字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """加载数据"""
    logger.info("Loading data...")
    
    # 加载预处理后的特征数据 (parquet格式)
    air_train = pd.read_parquet('data/outputs/air_train.parquet')
    hpg_train = pd.read_parquet('data/outputs/hpg_train.parquet')
    hpg_test = pd.read_parquet('data/outputs/hpg_test.parquet')
    
    logger.info(f"Data loaded. AIR train: {len(air_train)}, HPG train: {len(hpg_train)}")
    
    return air_train, hpg_train, hpg_test


def prepare_features(train_data, test_data):
    """准备特征"""
    exclude_cols = ['visitors', 'visit_date', 'store_id', 'day_gap']
    x_cols = [col for col in train_data.columns if col not in exclude_cols and train_data[col].dtype in ['int64', 'float64', 'bool']]
    y_cols = ['visitors']
    
    logger.info(f"Feature columns count: {len(x_cols)}")
    
    return x_cols, y_cols


def visualize_predictions_vs_actual(results, title, filename):
    """可视化预测值与实际值的对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 如果结果中有预测值和实际值
    if 'predictions' in results and 'actual' in results:
        predictions = results['predictions']
        actual = results['actual']
        
        # 绘制散点图
        ax.scatter(actual, predictions, alpha=0.5)
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{title} - Predictions vs Actual')
        
        # 计算并显示R²
        correlation = np.corrcoef(actual, predictions)[0, 1]
        ax.text(0.05, 0.95, f'R² = {correlation**2:.4f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'experiments/visualization_{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_results_comparison(all_results):
    """绘制所有实验结果的对比图"""
    if not all_results:
        return
    
    # 准备数据
    methods = list(all_results.keys())
    rmsle_values = list(all_results.values())
    
    # 创建柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(methods)), rmsle_values, color='skyblue')
    plt.xlabel('Methods')
    plt.ylabel('RMSLE')
    plt.title('Transfer Learning Methods Comparison')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    
    # 在柱子上添加数值标签
    for bar, value in zip(bars, rmsle_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('experiments/experiment_results_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_baseline_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行基线实验"""
    logger.info("=" * 60)
    logger.info("Running Baseline Experiment")
    logger.info("=" * 60)
    
    try:
        model_base = ModelBase(device="gpu")
        
        # 准备数据
        train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
            train_data=hpg_train,
            test_data=hpg_test,
            x_cols=x_cols,
            y_cols=y_cols,
            weight_col="day_gap",
            valid_days=7,
        )
        
        # 训练模型
        model = model_base.train_or_load_lgbm(
            train_matrix,
            valid_matrix,
            None,
            num_threads=4,
            seed=2018,
        )
        
        # 评估
        predictions = model.predict(te_x, num_iteration=model.best_iteration)
        rmsle_score = np.sqrt(mean_squared_error(te_y, predictions))
        
        results = {
            "model": model,
            "predictions": predictions,
            "actual": te_y,
            "rmsle": rmsle_score,
        }
        
        logger.info(f"Baseline RMSLE: {rmsle_score:.4f}")
        
        # 可视化结果
        visualize_predictions_vs_actual(results, "Baseline", "baseline")
        
        return results
    except Exception as e:
        logger.error(f"Error in baseline experiment: {e}")
        return None


def run_pseudo_labeling_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行伪标签迁移实验"""
    logger.info("=" * 60)
    logger.info("Running Pseudo Labeling Experiment")
    logger.info("=" * 60)
    
    try:
        results = pseudo_label_transfer_learning(
            source_data=air_train,
            target_data=hpg_train,
            test_data=hpg_test,
            x_cols=x_cols,
            y_cols=y_cols,
            source_model_path="lgbm_weights/air_model.pkl",
            pseudo_model_path=None,
            weight_col="day_gap",
            valid_days=7,
            device="gpu",
            seed=2018
        )
        
        logger.info(f"Pseudo Labeling RMSLE: {results['rmsle']:.4f}")
        
        # 可视化结果
        visualize_predictions_vs_actual(results, "Pseudo Labeling", "pseudo_labeling")
        
        return results
    except Exception as e:
        logger.error(f"Error in pseudo labeling experiment: {e}")
        return None


def run_fine_tune_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols, lr=0.01, rounds=1000):
    """运行微调实验"""
    logger.info("=" * 60)
    logger.info(f"Running Fine-tune Experiment with lr={lr}, rounds={rounds}")
    logger.info("=" * 60)
    
    try:
        results = fine_tune_transfer_learning(
            source_data=air_train,
            target_data=hpg_train,
            test_data=hpg_test,
            x_cols=x_cols,
            y_cols=y_cols,
            source_model_path="lgbm_weights/air_model.pkl",
            fine_tune_lr=lr,
            weight_col="day_gap",
            valid_days=7,
            num_round=rounds,
            device="gpu",
            seed=2018
        )
        
        logger.info(f"Fine-tune RMSLE: {results['rmsle']:.4f}")
        
        # 可视化结果
        visualize_predictions_vs_actual(results, f"Fine-tune (lr={lr})", f"fine_tune_lr_{lr}")
        
        return results
    except Exception as e:
        logger.error(f"Error in fine-tune experiment with lr={lr}, rounds={rounds}: {e}")
        return None


def run_adversarial_domain_adaptation_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行对抗域适应实验"""
    logger.info("=" * 60)
    logger.info("Running Adversarial Domain Adaptation Experiment")
    logger.info("=" * 60)
    
    try:
        results = adversarial_domain_adaptation(
            source_data=air_train,
            target_data=hpg_train,
            test_data=hpg_test,
            x_cols=x_cols,
            y_cols=y_cols,
            source_model_path="lgbm_weights/air_model.pkl",
            weight_col="day_gap",
            valid_days=7,
            num_round=1000,
            num_threads=4,
            seed=2018,
            device="gpu"
        )
        
        logger.info(f"Adversarial Domain Adaptation RMSLE: {results['rmsle']:.4f}")
        
        # 可视化结果
        visualize_predictions_vs_actual(results, "Adversarial Domain Adaptation", "adversarial_da")
        
        return results
    except Exception as e:
        logger.error(f"Error in adversarial domain adaptation experiment: {e}")
        return None


def run_super_ensemble_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols, topk=30, lr1=0.01, lr2=0.002):
    """运行超级集成实验"""
    logger.info("=" * 60)
    logger.info(f"Running Super Ensemble Experiment with topk={topk}, lr1={lr1}, lr2={lr2}")
    logger.info("=" * 60)
    
    try:
        results = super_ensemble_transfer_learning(
            source_data=air_train,
            target_data=hpg_train,
            test_data=hpg_test,
            x_cols=x_cols,
            y_cols=y_cols,
            source_model_path="lgbm_weights/air_model.pkl",
            weight_col="day_gap",
            valid_days=7,
            learning_rate_phase1=lr1,
            learning_rate_phase2=lr2,
            num_round_phase1=500,
            num_round_phase2=1000,
            top_k_features=topk,
            device="gpu",
            seed=2018
        )
        
        logger.info(f"Super Ensemble RMSLE: {results['rmsle']:.4f}")
        
        # 可视化结果
        visualize_predictions_vs_actual(results, f"Super Ensemble (topk={topk})", f"super_ensemble_topk_{topk}")
        
        return results
    except Exception as e:
        logger.error(f"Error in super ensemble experiment with topk={topk}, lr1={lr1}, lr2={lr2}: {e}")
        return None


def run_all_experiments():
    """运行所有实验"""
    logger.info("=" * 60)
    logger.info("Starting All Transfer Learning Experiments")
    logger.info("=" * 60)
    
    try:
        # 加载数据
        air_train, hpg_train, hpg_test = load_data()
        
        # 准备特征
        x_cols, y_cols = prepare_features(hpg_train, hpg_test)
        
        # 存储所有实验结果
        all_results = {}
        
        # 运行基线实验
        logger.info("Starting Baseline Experiment")
        baseline_result = run_baseline_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
        if baseline_result:
            all_results['Baseline'] = baseline_result['rmsle']
        
        # 运行伪标签实验
        logger.info("Starting Pseudo Labeling Experiment")
        pseudo_result = run_pseudo_labeling_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
        if pseudo_result:
            all_results['Pseudo Labeling'] = pseudo_result['rmsle']
        
        # 运行微调实验 (默认参数)
        logger.info("Starting Fine-tune Experiment")
        finetune_result = run_fine_tune_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
        if finetune_result:
            all_results['Fine-tune'] = finetune_result['rmsle']
        
        # 运行对抗域适应实验
        logger.info("Starting Adversarial Domain Adaptation Experiment")
        adversarial_result = run_adversarial_domain_adaptation_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
        if adversarial_result:
            all_results['Adversarial DA'] = adversarial_result['rmsle']
        
        # 运行超级集成实验 (默认参数)
        logger.info("Starting Super Ensemble Experiment")
        super_ensemble_result = run_super_ensemble_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
        if super_ensemble_result:
            all_results['Super Ensemble'] = super_ensemble_result['rmsle']
        
        # 输出所有结果摘要
        logger.info("=" * 60)
        logger.info("ALL EXPERIMENTS RESULTS SUMMARY")
        logger.info("=" * 60)
        for method, rmsle in sorted(all_results.items(), key=lambda item: item[1]):
            logger.info(f"{method}: {rmsle:.4f}")
        logger.info("=" * 60)
        
        # 绘制结果对比图
        plot_results_comparison(all_results)
        
        # 保存结果到JSON文件
        with open('experiments/results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info("All experiments completed. Results saved to experiments/results.json")
        logger.info("Visualizations saved in experiments/ directory.")
        
        return all_results
        
    except Exception as e:
        logger.error(f"Error during experiments: {e}")
        raise


def run_parameter_tuning_experiments():
    """运行参数调优实验"""
    logger.info("=" * 60)
    logger.info("Starting Parameter Tuning Experiments")
    logger.info("=" * 60)
    
    try:
        # 加载数据
        air_train, hpg_train, hpg_test = load_data()
        
        # 准备特征
        x_cols, y_cols = prepare_features(hpg_train, hpg_test)
        
        # 存储所有实验结果
        tuning_results = {}
        
        # 微调参数调优
        logger.info("Starting Fine-tune Parameter Tuning")
        finetune_params = [
            {'lr': 0.001, 'rounds': 1000},
            {'lr': 0.01, 'rounds': 1000},
            {'lr': 0.05, 'rounds': 1000},
        ]
        
        for params in finetune_params:
            lr = params['lr']
            rounds = params['rounds']
            result = run_fine_tune_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols, lr, rounds)
            if result:
                tuning_results[f'Fine-tune_lr_{lr}_rounds_{rounds}'] = result['rmsle']
        
        # 超级集成参数调优
        logger.info("Starting Super Ensemble Parameter Tuning")
        super_ensemble_params = [
            {'topk': 10, 'lr1': 0.01, 'lr2': 0.002},
            {'topk': 30, 'lr1': 0.01, 'lr2': 0.002},
            {'topk': 50, 'lr1': 0.01, 'lr2': 0.002},
        ]
        
        for params in super_ensemble_params:
            topk = params['topk']
            lr1 = params['lr1']
            lr2 = params['lr2']
            result = run_super_ensemble_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols, topk, lr1, lr2)
            if result:
                tuning_results[f'Super_Ensemble_topk_{topk}_lr1_{lr1}_lr2_{lr2}'] = result['rmsle']
        
        # 输出调优结果摘要
        logger.info("=" * 60)
        logger.info("PARAMETER TUNING RESULTS SUMMARY")
        logger.info("=" * 60)
        for method, rmsle in sorted(tuning_results.items(), key=lambda item: item[1]):
            logger.info(f"{method}: {rmsle:.4f}")
        logger.info("=" * 60)
        
        # 保存调优结果到JSON文件
        with open('experiments/tuning_results.json', 'w') as f:
            json.dump(tuning_results, f, indent=2)
        
        logger.info("Parameter tuning experiments completed. Results saved to experiments/tuning_results.json")
        
        return tuning_results
        
    except Exception as e:
        logger.error(f"Error during parameter tuning experiments: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run transfer learning experiments')
    parser.add_argument('--experiment', choices=['all', 'tuning', 'baseline', 'pseudo', 'finetune', 'adversarial', 'super_ensemble'],
                        default='all', help='Which experiment to run')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for fine-tune experiment')
    parser.add_argument('--rounds', type=int, default=1000, help='Number of rounds for fine-tune experiment')
    parser.add_argument('--topk', type=int, default=30, help='Top K features for super ensemble experiment')
    parser.add_argument('--lr1', type=float, default=0.01, help='Learning rate phase 1 for super ensemble')
    parser.add_argument('--lr2', type=float, default=0.002, help='Learning rate phase 2 for super ensemble')
    
    args = parser.parse_args()
    
    if args.experiment == 'all':
        run_all_experiments()
    elif args.experiment == 'tuning':
        run_parameter_tuning_experiments()
    elif args.experiment == 'baseline':
        air_train, hpg_train, hpg_test = load_data()
        x_cols, y_cols = prepare_features(hpg_train, hpg_test)
        run_baseline_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
    elif args.experiment == 'pseudo':
        air_train, hpg_train, hpg_test = load_data()
        x_cols, y_cols = prepare_features(hpg_train, hpg_test)
        run_pseudo_labeling_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
    elif args.experiment == 'finetune':
        air_train, hpg_train, hpg_test = load_data()
        x_cols, y_cols = prepare_features(hpg_train, hpg_test)
        run_fine_tune_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols, args.lr, args.rounds)
    elif args.experiment == 'adversarial':
        air_train, hpg_train, hpg_test = load_data()
        x_cols, y_cols = prepare_features(hpg_train, hpg_test)
        run_adversarial_domain_adaptation_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
    elif args.experiment == 'super_ensemble':
        air_train, hpg_train, hpg_test = load_data()
        x_cols, y_cols = prepare_features(hpg_train, hpg_test)
        run_super_ensemble_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols, args.topk, args.lr1, args.lr2)


if __name__ == "__main__":
    main()