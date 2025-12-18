#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行修改后的Super Ensemble方法并记录详细日志
"""

import pandas as pd
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from src.model.super_ensemble import super_ensemble_transfer_learning

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('super_ensemble_detailed.log'),
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

def main():
    """主函数"""
    logger.info("="*60)
    logger.info("Starting Detailed Super Ensemble Transfer Learning Test")
    logger.info("="*60)
    
    # 加载数据
    air_train, hpg_train, hpg_test, x_cols, y_cols = load_data()
    
    # 运行超级集成迁移学习
    try:
        results = super_ensemble_transfer_learning(
            source_data=air_train,
            target_data=hpg_train,
            test_data=hpg_test,
            x_cols=x_cols,
            y_cols=y_cols,
            source_model_path="./lgbm_weights/air_model.pkl",
            super_ensemble_model_path="./lgbm_weights/super_ensemble_model.pkl",
            weight_col="day_gap",
            valid_days=7,
            learning_rate_phase1=0.01,
            learning_rate_phase2=0.002,
            num_round_phase1=500,
            num_round_phase2=3000,
            num_threads=4,
            seed=2018,
            device="gpu",
        )
        
        logger.info("="*60)
        logger.info("FINAL RESULTS")
        logger.info("="*60)
        logger.info(f"Final RMSLE: {results['rmsle']:.4f}")
        logger.info("="*60)
        logger.info("Detailed Super Ensemble test completed. Results saved to super_ensemble_detailed.log")
        
    except Exception as e:
        logger.error(f"Error during Super Ensemble transfer learning: {e}")
        raise

if __name__ == "__main__":
    main()