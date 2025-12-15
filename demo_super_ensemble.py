#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超级集成迁移学习方法演示脚本

展示基于单模型的超级集成迁移学习方法及其相对于基线的改进
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb

from src.model.base import ModelBase
from src.model import super_ensemble_transfer_learning


def main():
    """主函数"""
    print("=" * 70)
    print("超级集成迁移学习方法演示")
    print("=" * 70)
    print("方法概要:")
    print("  单一模型方法，结合以下技术：")
    print("    1. 智能样本筛选 - 选择源域中与目标域最相关的样本")
    print("    2. 自适应特征对齐 - 只对相关特征进行对齐")
    print("    3. 两阶段训练策略 - 先快速收敛，再精细调优")
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
    print(f"RMSLE: {baseline_results['rmsle']:.4f}")
    
    # 超级集成迁移学习模型
    print("\n" + "-" * 50)
    print("超级集成迁移学习模型")
    print("-" * 50)
    
    try:
        super_ensemble_results = super_ensemble_transfer_learning(
            source_data=air_train,
            target_data=hpg_train,
            test_data=hpg_test,
            x_cols=x_cols,
            y_cols=y_cols,
            source_model_path='./lgbm_weights/air_model.pkl',
            super_ensemble_model_path='./lgbm_weights/super_ensemble_model.pkl',
            device=device
        )
        print(f"RMSLE: {super_ensemble_results['rmsle']:.4f}")
        
        # 计算改进
        baseline_rmsle = baseline_results['rmsle']
        super_ensemble_rmsle = super_ensemble_results['rmsle']
        improvement = baseline_rmsle - super_ensemble_rmsle
        improvement_percentage = (improvement / baseline_rmsle) * 100
        
        print(f"\n效果对比:")
        print(f"  - 基线模型 RMSLE:              {baseline_rmsle:.4f}")
        print(f"  - 超级集成迁移学习 RMSLE:       {super_ensemble_rmsle:.4f}")
        print(f"  - 绝对改进:                    {improvement:+.4f}")
        print(f"  - 相对改进:                    {improvement_percentage:+.2f}%")
        
        if improvement > 0:
            print(f"\n🎉 迁移学习带来了显著的性能提升!")
            print(f"   通过有效地利用源域(AIR)的知识，我们在目标域(HPG)上获得了 {improvement_percentage:.2f}% 的性能提升。")
        else:
            print(f"\n⚠️  迁移学习效果不佳，请检查方法或参数。")
            
    except Exception as e:
        print(f"超级集成迁移学习失败: {e}")
        print("请检查模型路径和数据是否正确。")
    
    print("\n" + "=" * 70)
    print("演示结束")
    print("=" * 70)


if __name__ == "__main__":
    main()