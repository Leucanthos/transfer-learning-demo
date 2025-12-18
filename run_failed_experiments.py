import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.transfer import pseudo_label_transfer_learning
from src.model.advanced_transfer import adversarial_domain_adaptation

# 设置随机种子
np.random.seed(2018)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("failed_experiments.log"),
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

def run_pseudo_labeling_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行伪标签迁移实验"""
    logger.info("=" * 60)
    logger.info("Running Pseudo Labeling Experiment")
    logger.info("=" * 60)
    
    # 重新加载数据以获得完整的特征集合
    air_train_full = pd.read_parquet('data/outputs/air_train.parquet')
    hpg_train_full = pd.read_parquet('data/outputs/hpg_train.parquet')
    hpg_test_full = pd.read_parquet('data/outputs/hpg_test.parquet')
    
    # 获取完整的特征列（包括非数值类型）
    exclude_cols = ['visitors', 'visit_date', 'store_id', 'day_gap']
    x_cols_full = [col for col in hpg_train_full.columns if col not in exclude_cols and hpg_train_full[col].dtype in ['int64', 'float64', 'bool']]
    
    results = pseudo_label_transfer_learning(
        source_data=air_train_full,
        target_data=hpg_train_full,
        test_data=hpg_test_full,
        x_cols=x_cols_full,
        y_cols=y_cols,
        source_model_path="lgbm_weights/air_model.pkl",
        pseudo_model_path=None,
        weight_col="day_gap",
        valid_days=7,
        device="gpu",
        seed=2018
    )
    
    logger.info(f"Pseudo Labeling RMSLE: {results['rmsle']:.4f}")
    return results


def run_adversarial_domain_adaptation_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行对抗域适应实验"""
    logger.info("=" * 60)
    logger.info("Running Adversarial Domain Adaptation Experiment")
    logger.info("=" * 60)
    
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
    return results

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Starting Failed Transfer Learning Experiments")
    logger.info("=" * 60)
    
    try:
        # 加载数据
        air_train, hpg_train, hpg_test = load_data()
        
        # 准备特征
        x_cols, y_cols = prepare_features(hpg_train, hpg_test)
        
        # 存储所有实验结果
        all_results = {}
        
        # 运行伪标签实验
        try:
            pseudo_result = run_pseudo_labeling_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
            all_results['pseudo_labeling'] = pseudo_result['rmsle']
        except Exception as e:
            logger.error(f"Error in pseudo labeling experiment: {e}")
        
        # 运行对抗域适应实验
        try:
            adversarial_result = run_adversarial_domain_adaptation_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
            all_results['adversarial_domain_adaptation'] = adversarial_result['rmsle']
        except Exception as e:
            logger.error(f"Error in adversarial domain adaptation experiment: {e}")
        
        # 输出所有结果摘要
        logger.info("=" * 60)
        logger.info("FAILED EXPERIMENTS RESULTS SUMMARY")
        logger.info("=" * 60)
        for method, rmsle in all_results.items():
            logger.info(f"{method}: {rmsle:.4f}")
        logger.info("=" * 60)
        logger.info("Failed experiments completed. Results saved to failed_experiments.log")
        
    except Exception as e:
        logger.error(f"Error during experiments: {e}")
        raise

if __name__ == "__main__":
    main()