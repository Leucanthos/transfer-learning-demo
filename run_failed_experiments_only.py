import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.transfer import pseudo_label_transfer_learning, fine_tune_transfer_learning
from src.model.advanced_transfer import adversarial_domain_adaptation
from src.model.baseline import evaluate_model  # 导入evaluate_model函数
from src.model.base import ModelBase

# 设置随机种子
np.random.seed(2018)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("failed_experiments_only.log"),
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
    
    # 解决特征数量不匹配问题
    try:
        # 加载源模型以检查其期望的特征名称和数量
        with open("lgbm_weights/air_model.pkl", "rb") as f:
            source_model = pickle.load(f)
        
        # 获取源模型的特征名称
        expected_feature_names = source_model.feature_name()
        expected_feature_count = len(expected_feature_names)
        logger.info(f"Source model expects {expected_feature_count} features: {expected_feature_names[:5]}...")
        
        # 确定哪些特征是分类特征（非数值型）
        categorical_features = ['weekday', 'air_genre_name', 'air_area_name', 'holiday']
        numeric_features = [col for col in expected_feature_names if col not in categorical_features]
        
        logger.info(f"Categorical features: {categorical_features}")
        logger.info(f"Numeric features: {len(numeric_features)}")
        
        # 检查所有需要的特征是否都存在
        missing_in_air = set(expected_feature_names) - set(air_train.columns)
        missing_in_hpg_train = set(expected_feature_names) - set(hpg_train.columns)
        missing_in_hpg_test = set(expected_feature_names) - set(hpg_test.columns)
        
        if missing_in_air:
            logger.warning(f"Missing columns in air_train: {missing_in_air}")
        if missing_in_hpg_train:
            logger.warning(f"Missing columns in hpg_train: {missing_in_hpg_train}")
        if missing_in_hpg_test:
            logger.warning(f"Missing columns in hpg_test: {missing_in_hpg_test}")
        
        # 对分类特征进行简单编码处理
        air_train_processed = air_train.copy()
        hpg_train_processed = hpg_train.copy()
        hpg_test_processed = hpg_test.copy()
        
        # 对每个分类特征进行Label Encoding
        from sklearn.preprocessing import LabelEncoder
        
        for cat_feature in categorical_features:
            if cat_feature in expected_feature_names:
                # 创建标签编码器
                le = LabelEncoder()
                
                # 合并所有数据以确保编码一致
                if cat_feature in air_train.columns and cat_feature in hpg_train.columns and cat_feature in hpg_test.columns:
                    all_values = pd.concat([
                        air_train[cat_feature].astype(str),
                        hpg_train[cat_feature].astype(str),
                        hpg_test[cat_feature].astype(str)
                    ]).unique()
                    
                    # 拟合编码器
                    le.fit(all_values)
                    
                    # 转换各个数据集
                    air_train_processed[cat_feature] = le.transform(air_train[cat_feature].astype(str))
                    hpg_train_processed[cat_feature] = le.transform(hpg_train[cat_feature].astype(str))
                    hpg_test_processed[cat_feature] = le.transform(hpg_test[cat_feature].astype(str))
                    
                    logger.info(f"Encoded categorical feature '{cat_feature}' with {len(all_values)} unique values")
        
        results = pseudo_label_transfer_learning(
            source_data=air_train_processed,
            target_data=hpg_train_processed,
            test_data=hpg_test_processed,
            x_cols=expected_feature_names,  # 使用完整的特征列表
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
    except Exception as e:
        logger.error(f"Error in pseudo labeling experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_adversarial_domain_adaptation_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行对抗域适应实验"""
    logger.info("=" * 60)
    logger.info("Running Adversarial Domain Adaptation Experiment")
    logger.info("=" * 60)
    
    try:
        # 创建ModelBase实例以便使用evaluate_model方法
        model_base = ModelBase(device="gpu")
        
        # 定义本地evaluate_model函数，使用model_base的方法
        def local_evaluate_model(model, te_x, te_y):
            return model_base.evaluate_model(model, te_x, te_y)
        
        # 替换advanced_transfer模块中的evaluate_model函数
        import src.model.advanced_transfer as adv_transfer
        adv_transfer.evaluate_model = local_evaluate_model
        
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
    except Exception as e:
        logger.error(f"Error in adversarial domain adaptation experiment: {e}")
        return None

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Starting Failed Transfer Learning Experiments Only")
    logger.info("=" * 60)
    
    try:
        # 加载数据
        air_train, hpg_train, hpg_test = load_data()
        
        # 准备特征
        x_cols, y_cols = prepare_features(hpg_train, hpg_test)
        
        # 存储所有实验结果
        all_results = {}
        
        # 运行伪标签实验
        logger.info("Starting Pseudo Labeling Experiment")
        pseudo_result = run_pseudo_labeling_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
        if pseudo_result:
            all_results['pseudo_labeling'] = pseudo_result['rmsle']
        
        # 运行对抗域适应实验
        logger.info("Starting Adversarial Domain Adaptation Experiment")
        adversarial_result = run_adversarial_domain_adaptation_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
        if adversarial_result:
            all_results['adversarial_domain_adaptation'] = adversarial_result['rmsle']
        
        # 输出所有结果摘要
        logger.info("=" * 60)
        logger.info("FAILED EXPERIMENTS ONLY RESULTS SUMMARY")
        logger.info("=" * 60)
        for method, rmsle in all_results.items():
            logger.info(f"{method}: {rmsle:.4f}")
        logger.info("=" * 60)
        logger.info("Failed experiments completed. Results saved to failed_experiments_only.log")
        
    except Exception as e:
        logger.error(f"Error during experiments: {e}")
        raise

if __name__ == "__main__":
    main()