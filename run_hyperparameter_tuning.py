import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.baseline import model_base
from src.model.super_ensemble import super_ensemble_transfer_learning
from src.model.transfer import fine_tune_transfer_learning

# 设置随机种子
np.random.seed(2018)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperparameter_tuning.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

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

def run_fine_tune_experiments(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行不同参数的微调实验"""
    logger.info("=" * 60)
    logger.info("Running Fine-tune Experiments with Different Parameters")
    logger.info("=" * 60)
    
    results = {}
    
    # 不同的学习率
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    for lr in learning_rates:
        logger.info(f"Running Fine-tune with learning rate: {lr}")
        try:
            result = fine_tune_transfer_learning(
                source_data=air_train,
                target_data=hpg_train,
                test_data=hpg_test,
                x_cols=x_cols,
                y_cols=y_cols,
                source_model_path="lgbm_weights/air_model.pkl",
                target_model_path=None,
                fine_tune_lr=lr,
                weight_col="day_gap",
                valid_days=7,
                device="gpu",
                seed=2018
            )
            results[f'fine_tune_lr_{lr}'] = result['rmsle']
            logger.info(f"Fine-tune with learning rate {lr} - RMSLE: {result['rmsle']:.4f}")
        except Exception as e:
            logger.error(f"Error in fine-tune experiment with lr={lr}: {e}")
    
    return results

def run_super_ensemble_experiments(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行不同参数的超级集成实验"""
    logger.info("=" * 60)
    logger.info("Running Super Ensemble Experiments with Different Parameters")
    logger.info("=" * 60)
    
    results = {}
    
    # 不同的特征选择比例
    top_k_ratios = [0.1, 0.2, 0.3, 0.5]
    
    for ratio in top_k_ratios:
        k = int(len(x_cols) * ratio)
        logger.info(f"Running Super Ensemble with top {k} features ({ratio*100}%)")
        try:
            result = super_ensemble_transfer_learning(
                source_data=air_train,
                target_data=hpg_train,
                test_data=hpg_test,
                x_cols=x_cols,
                y_cols=y_cols,
                source_model_path="lgbm_weights/air_model.pkl",
                target_model_path=None,
                weight_col="day_gap",
                valid_days=7,
                topk=k,
                device="gpu",
                seed=2018
            )
            results[f'super_ensemble_top_{k}'] = result['rmsle']
            logger.info(f"Super Ensemble with top {k} features - RMSLE: {result['rmsle']:.4f}")
        except Exception as e:
            logger.error(f"Error in super ensemble experiment with topk={k}: {e}")
    
    return results

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Starting Hyperparameter Tuning Experiments")
    logger.info("=" * 60)
    
    try:
        # 加载数据
        air_train, hpg_train, hpg_test = load_data()
        
        # 准备特征
        x_cols, y_cols = prepare_features(hpg_train, hpg_test)
        
        # 存储所有实验结果
        all_results = {}
        
        # 运行微调实验
        fine_tune_results = run_fine_tune_experiments(air_train, hpg_train, hpg_test, x_cols, y_cols)
        all_results.update(fine_tune_results)
        
        # 运行超级集成实验
        super_ensemble_results = run_super_ensemble_experiments(air_train, hpg_train, hpg_test, x_cols, y_cols)
        all_results.update(super_ensemble_results)
        
        # 输出所有结果摘要
        logger.info("=" * 60)
        logger.info("HYPERPARAMETER TUNING RESULTS SUMMARY")
        logger.info("=" * 60)
        for method, rmsle in all_results.items():
            logger.info(f"{method}: {rmsle:.4f}")
        logger.info("=" * 60)
        logger.info("Hyperparameter tuning completed. Results saved to hyperparameter_tuning.log")
        
    except Exception as e:
        logger.error(f"Error during experiments: {e}")
        raise

if __name__ == "__main__":
    main()