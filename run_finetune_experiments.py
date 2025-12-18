import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.transfer import fine_tune_transfer_learning
from src.model.super_ensemble import super_ensemble_transfer_learning

# 设置随机种子
np.random.seed(2018)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("finetune_experiments.log"),
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
    
    # 不同的学习率和迭代次数组合
    params_combinations = [
        {'lr': 0.001, 'rounds': 1000},
        {'lr': 0.001, 'rounds': 2000},
        {'lr': 0.005, 'rounds': 1000},
        {'lr': 0.005, 'rounds': 2000},
        {'lr': 0.01, 'rounds': 1000},
        {'lr': 0.01, 'rounds': 2000},
        {'lr': 0.05, 'rounds': 1000},
        {'lr': 0.05, 'rounds': 2000},
        {'lr': 0.1, 'rounds': 1000},
        {'lr': 0.1, 'rounds': 2000},
    ]
    
    for params in params_combinations:
        lr = params['lr']
        rounds = params['rounds']
        exp_name = f'fine_tune_lr_{lr}_rounds_{rounds}'
        
        logger.info(f"Running Fine-tune with learning rate: {lr}, rounds: {rounds}")
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
                seed=2018,
                num_round=rounds
            )
            results[exp_name] = result['rmsle']
            logger.info(f"Fine-tune with learning rate {lr}, rounds {rounds} - RMSLE: {result['rmsle']:.4f}")
        except Exception as e:
            logger.error(f"Error in fine-tune experiment with lr={lr}, rounds={rounds}: {e}")
    
    return results

def run_super_ensemble_finetune_experiments(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行超级集成微调实验"""
    logger.info("=" * 60)
    logger.info("Running Super Ensemble Fine-tune Experiments")
    logger.info("=" * 60)
    
    results = {}
    
    # 不同的topk特征选择和学习率组合
    params_combinations = [
        {'topk': 10, 'lr_phase1': 0.01, 'lr_phase2': 0.002},
        {'topk': 10, 'lr_phase1': 0.05, 'lr_phase2': 0.005},
        {'topk': 30, 'lr_phase1': 0.01, 'lr_phase2': 0.002},
        {'topk': 30, 'lr_phase1': 0.05, 'lr_phase2': 0.005},
        {'topk': 50, 'lr_phase1': 0.01, 'lr_phase2': 0.002},
        {'topk': 50, 'lr_phase1': 0.05, 'lr_phase2': 0.005},
        {'topk': 100, 'lr_phase1': 0.01, 'lr_phase2': 0.002},
        {'topk': 100, 'lr_phase1': 0.05, 'lr_phase2': 0.005},
    ]
    
    for params in params_combinations:
        topk = params['topk']
        lr_phase1 = params['lr_phase1']
        lr_phase2 = params['lr_phase2']
        exp_name = f'super_ensemble_topk_{topk}_lr1_{lr_phase1}_lr2_{lr_phase2}'
        
        logger.info(f"Running Super Ensemble with topk: {topk}, learning rate phase1: {lr_phase1}, phase2: {lr_phase2}")
        try:
            result = super_ensemble_transfer_learning(
                source_data=air_train,
                target_data=hpg_train,
                test_data=hpg_test,
                x_cols=x_cols,
                y_cols=y_cols,
                source_model_path="lgbm_weights/air_model.pkl",
                super_ensemble_model_path=None,
                weight_col="day_gap",
                valid_days=7,
                learning_rate_phase1=lr_phase1,
                learning_rate_phase2=lr_phase2,
                num_round_phase1=500,
                num_round_phase2=3000,
                device="gpu",
                seed=2018
            )
            results[exp_name] = result['rmsle']
            logger.info(f"Super Ensemble with topk {topk}, learning rate phase1 {lr_phase1}, phase2 {lr_phase2} - RMSLE: {result['rmsle']:.4f}")
        except Exception as e:
            logger.error(f"Error in super ensemble experiment with topk={topk}, lr_phase1={lr_phase1}, lr_phase2={lr_phase2}: {e}")
    
    return results

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Starting Fine-tune Experiments")
    logger.info("=" * 60)
    
    try:
        # 加载数据
        air_train, hpg_train, hpg_test = load_data()
        
        # 准备特征
        x_cols, y_cols = prepare_features(hpg_train, hpg_test)
        
        # 存储所有实验结果
        all_results = {}
        
        # 运行微调实验
        finetune_results = run_fine_tune_experiments(air_train, hpg_train, hpg_test, x_cols, y_cols)
        all_results.update(finetune_results)
        
        # 运行超级集成微调实验
        super_ensemble_results = run_super_ensemble_finetune_experiments(air_train, hpg_train, hpg_test, x_cols, y_cols)
        all_results.update(super_ensemble_results)
        
        # 输出所有结果摘要
        logger.info("=" * 60)
        logger.info("FINE-TUNE EXPERIMENTS RESULTS SUMMARY")
        logger.info("=" * 60)
        for method, rmsle in sorted(all_results.items(), key=lambda item: item[1]):
            logger.info(f"{method}: {rmsle:.4f}")
        logger.info("=" * 60)
        logger.info("Fine-tune experiments completed. Results saved to finetune_experiments.log")
        
    except Exception as e:
        logger.error(f"Error during experiments: {e}")
        raise

if __name__ == "__main__":
    main()