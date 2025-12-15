#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数调优实验脚本
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb

from src.model.base import ModelBase
from src.model.super_ensemble import super_ensemble_transfer_learning


def run_hyperparameter_experiment(name, params, air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行单次超参数实验"""
    print(f"\n{'='*50}")
    print(f"实验: {name}")
    print(f"参数: {params}")
    print(f"{'='*50}")
    
    try:
        results = super_ensemble_transfer_learning(
            source_data=air_train,
            target_data=hpg_train,
            test_data=hpg_test,
            x_cols=x_cols,
            y_cols=y_cols,
            source_model_path='./lgbm_weights/air_model.pkl',
            super_ensemble_model_path=None,  # 不保存中间模型
            learning_rate_phase1=params['learning_rate_phase1'],
            learning_rate_phase2=params['learning_rate_phase2'],
            num_round_phase1=params['num_round_phase1'],
            num_round_phase2=params['num_round_phase2'],
            device=params['device']
        )
        print(f"RMSLE: {results['rmsle']:.4f}")
        return results['rmsle']
    except Exception as e:
        print(f"实验失败: {e}")
        return float('inf')


def main():
    """主函数"""
    print("=" * 70)
    print("超参数调优实验")
    print("=" * 70)
    
    # 数据路径
    OUTPUTS = Path('./data/outputs')
    
    # 加载数据
    print("加载数据...")
    air_train = pd.read_parquet(OUTPUTS/"air_train.parquet")    # 源域数据 (AIR)
    hpg_train = pd.read_parquet(OUTPUTS/"hpg_train.parquet")    # 目标域训练数据 (HPG)
    hpg_test = pd.read_parquet(OUTPUTS/"hpg_test.parquet")      # 目标域测试数据 (HPG)
    
    # 特征列
    x_cols = hpg_train.drop(columns=["air_store_id","visit_date","visitors"]).columns.tolist()
    y_cols = ["visitors"]
    
    print(f"数据集信息:")
    print(f"  - 特征数量: {len(x_cols)}")
    print(f"  - AIR训练样本数: {len(air_train)}")
    print(f"  - HPG训练样本数: {len(hpg_train)}")
    print(f"  - HPG测试样本数: {len(hpg_test)}")
    
    # 检查是否有可用的GPU
    try:
        device = "gpu"
        # 简单测试GPU是否可用
        test_params = {'device_type': 'gpu', 'verbose': -1}
        train_data = lgb.Dataset(np.array([[1, 2], [3, 4]]), label=np.array([1, 2]))
        lgb.train(test_params, train_data, 1)
        print(f"  - 使用设备: GPU (可用)")
    except Exception as e:
        device = "cpu"
        print(f"  - 使用设备: CPU (GPU不可用: {e})")
    
    # 基线模型结果 (直接在HPG上训练)
    print("\n" + "-" * 50)
    print("基线模型 (仅使用HPG数据训练)")
    print("-" * 50)
    
    model_base = ModelBase(device=device)
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=hpg_train,
        test_data=hpg_test,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col='day_gap',
        valid_days=7
    )
    
    baseline_model = model_base.train_or_load_lgbm(
        train_matrix=train_matrix,
        valid_matrix=valid_matrix,
        model_path='./lgbm_weights/hpg_baseline_model.pkl'
    )
    
    baseline_results = model_base.evaluate_model(baseline_model, te_x, te_y)
    baseline_rmsle = baseline_results['rmsle']
    print(f"基线模型 RMSLE: {baseline_rmsle:.4f}")
    
    # 定义实验参数
    experiments = [
        {
            'name': '默认参数',
            'params': {
                'learning_rate_phase1': 0.01,
                'learning_rate_phase2': 0.002,
                'num_round_phase1': 500,
                'num_round_phase2': 3000,
                'device': device
            }
        },
        {
            'name': '调优版本1',
            'params': {
                'learning_rate_phase1': 0.007,
                'learning_rate_phase2': 0.001,
                'num_round_phase1': 800,
                'num_round_phase2': 2500,
                'device': device
            }
        },
        {
            'name': '调优版本2',
            'params': {
                'learning_rate_phase1': 0.005,
                'learning_rate_phase2': 0.001,
                'num_round_phase1': 1000,
                'num_round_phase2': 2000,
                'device': device
            }
        },
        {
            'name': '调优版本3',
            'params': {
                'learning_rate_phase1': 0.005,
                'learning_rate_phase2': 0.0005,
                'num_round_phase1': 800,
                'num_round_phase2': 2500,
                'device': device
            }
        },
        {
            'name': '调优版本4',
            'params': {
                'learning_rate_phase1': 0.003,
                'learning_rate_phase2': 0.001,
                'num_round_phase1': 1200,
                'num_round_phase2': 2000,
                'device': device
            }
        }
    ]
    
    # 运行实验
    results = []
    for exp in experiments:
        rmsle = run_hyperparameter_experiment(
            exp['name'], exp['params'], 
            air_train, hpg_train, hpg_test, 
            x_cols, y_cols
        )
        results.append({
            'name': exp['name'],
            'rmsle': rmsle,
            'improvement': baseline_rmsle - rmsle if rmsle != float('inf') else 0
        })
    
    # 输出结果汇总
    print(f"\n{'='*70}")
    print("实验结果汇总")
    print(f"{'='*70}")
    print(f"{'方法':<15} {'RMSLE':<10} {'相比基线提升':<15}")
    print("-" * 70)
    print(f"{'基线模型':<15} {baseline_rmsle:<10.4f} {'-':<15}")
    
    for result in results:
        improvement_str = f"{result['improvement']:+.2%}" if result['improvement'] != 0 else "-"
        print(f"{result['name']:<15} {result['rmsle']:<10.4f} {improvement_str:<15}")
    
    best_result = min(results, key=lambda x: x['rmsle'])
    print(f"\n最佳结果: {best_result['name']} (RMSLE: {best_result['rmsle']:.4f})")
    print(f"相比基线提升: {best_result['improvement']:+.2%}")
    
    print("\n" + "=" * 70)
    print("超参数调优实验结束")
    print("=" * 70)


if __name__ == "__main__":
    main()