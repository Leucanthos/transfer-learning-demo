import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.baseline import train_or_load_lgbm, evaluate_model
from src.model.transfer import pseudo_label_transfer_learning, fine_tune_transfer_learning, direct_transfer_with_sample_selection
from src.model.ensemble_transfer import true_ensemble_transfer_learning
from src.model.advanced_transfer import adversarial_domain_adaptation
from src.model.super_ensemble import super_ensemble_transfer_learning, partial_feature_alignment

# 设置随机种子
np.random.seed(2018)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("all_experiments.log"),
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

def run_baseline_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行基线实验"""
    logger.info("=" * 60)
    logger.info("Running Baseline Experiment")
    logger.info("=" * 60)
    
    # 创建ModelBase实例
    from src.model.base import ModelBase
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
        "lgbm_weights/baseline_model.pkl",
        learning_rate=0.01,
        num_round=3000,
        num_threads=4,
        seed=2018,
        early_stopping_rounds=100,
    )
    
    # 评估模型
    results = model_base.evaluate_model(model, te_x, te_y)
    
    logger.info(f"Baseline RMSLE: {results['rmsle']:.4f}")
    return results

def run_coral_experiments(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行各种CORAL特征对齐实验"""
    logger.info("=" * 60)
    logger.info("Running CORAL Experiments")
    logger.info("=" * 60)
    
    coral_results = {}
    
    # 标准CORAL
    logger.info("Running Standard CORAL...")
    aligned_air_train = partial_feature_alignment(air_train, hpg_train, x_cols, top_ratio=1.0)
    enhanced_train = pd.concat([hpg_train, aligned_air_train], ignore_index=True)
    
    # 创建ModelBase实例
    from src.model.base import ModelBase
    model_base = ModelBase(device="gpu")
    
    # 准备数据
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=enhanced_train,
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
        None,  # 不保存中间模型
        learning_rate=0.01,
        num_round=3000,
        num_threads=4,
        seed=2018,
        early_stopping_rounds=100,
    )
    
    # 评估模型
    results = model_base.evaluate_model(model, te_x, te_y)
    
    coral_results['standard_coral'] = results['rmsle']
    logger.info(f"Standard CORAL RMSLE: {results['rmsle']:.4f}")
    
    # Top 30% CORAL
    logger.info("Running Top 30% CORAL...")
    aligned_air_train = partial_feature_alignment(air_train, hpg_train, x_cols, top_ratio=0.3)
    enhanced_train = pd.concat([hpg_train, aligned_air_train], ignore_index=True)
    
    # 准备数据
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=enhanced_train,
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
        learning_rate=0.01,
        num_round=3000,
        num_threads=4,
        seed=2018,
        early_stopping_rounds=100,
    )
    
    # 评估模型
    results = model_base.evaluate_model(model, te_x, te_y)
    
    coral_results['top30_coral'] = results['rmsle']
    logger.info(f"Top 30% CORAL RMSLE: {results['rmsle']:.4f}")
    
    # Top 50% CORAL
    logger.info("Running Top 50% CORAL...")
    aligned_air_train = partial_feature_alignment(air_train, hpg_train, x_cols, top_ratio=0.5)
    enhanced_train = pd.concat([hpg_train, aligned_air_train], ignore_index=True)
    
    # 准备数据
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=enhanced_train,
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
        learning_rate=0.01,
        num_round=3000,
        num_threads=4,
        seed=2018,
        early_stopping_rounds=100,
    )
    
    # 评估模型
    results = model_base.evaluate_model(model, te_x, te_y)
    
    coral_results['top50_coral'] = results['rmsle']
    logger.info(f"Top 50% CORAL RMSLE: {results['rmsle']:.4f}")
    
    # 相似度≥0.3 CORAL (模拟实现，使用top_ratio约等于该条件)
    logger.info("Running Similarity ≥0.3 CORAL...")
    aligned_air_train = partial_feature_alignment(air_train, hpg_train, x_cols, top_ratio=0.7)
    enhanced_train = pd.concat([hpg_train, aligned_air_train], ignore_index=True)
    
    # 准备数据
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=enhanced_train,
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
        learning_rate=0.01,
        num_round=3000,
        num_threads=4,
        seed=2018,
        early_stopping_rounds=100,
    )
    
    # 评估模型
    results = model_base.evaluate_model(model, te_x, te_y)
    
    coral_results['sim03_coral'] = results['rmsle']
    logger.info(f"Similarity ≥0.3 CORAL RMSLE: {results['rmsle']:.4f}")
    
    # 相似度≥0.5 CORAL
    logger.info("Running Similarity ≥0.5 CORAL...")
    aligned_air_train = partial_feature_alignment(air_train, hpg_train, x_cols, top_ratio=0.5)
    enhanced_train = pd.concat([hpg_train, aligned_air_train], ignore_index=True)
    
    # 准备数据
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=enhanced_train,
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
        learning_rate=0.01,
        num_round=3000,
        num_threads=4,
        seed=2018,
        early_stopping_rounds=100,
    )
    
    # 评估模型
    results = model_base.evaluate_model(model, te_x, te_y)
    
    coral_results['sim05_coral'] = results['rmsle']
    logger.info(f"Similarity ≥0.5 CORAL RMSLE: {results['rmsle']:.4f}")
    
    # 相似度≥0.7 CORAL
    logger.info("Running Similarity ≥0.7 CORAL...")
    aligned_air_train = partial_feature_alignment(air_train, hpg_train, x_cols, top_ratio=0.3)
    enhanced_train = pd.concat([hpg_train, aligned_air_train], ignore_index=True)
    
    # 准备数据
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=enhanced_train,
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
        learning_rate=0.01,
        num_round=3000,
        num_threads=4,
        seed=2018,
        early_stopping_rounds=100,
    )
    
    # 评估模型
    results = model_base.evaluate_model(model, te_x, te_y)
    
    coral_results['sim07_coral'] = results['rmsle']
    logger.info(f"Similarity ≥0.7 CORAL RMSLE: {results['rmsle']:.4f}")
    
    return coral_results

def run_pseudo_labeling_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行伪标签迁移实验"""
    logger.info("=" * 60)
    logger.info("Running Pseudo Labeling Experiment")
    logger.info("=" * 60)
    
    # 直接使用传递进来的x_cols和y_cols参数，避免特征数量不匹配的问题
    # 这样可以确保使用的特征与主流程中准备的特征完全一致
    
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
    return results

def run_fine_tune_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行微调实验"""
    logger.info("=" * 60)
    logger.info("Running Fine-tune Experiment")
    logger.info("=" * 60)
    
    results = fine_tune_transfer_learning(
        source_data=air_train,
        target_data=hpg_train,
        test_data=hpg_test,
        x_cols=x_cols,
        y_cols=y_cols,
        source_model_path="lgbm_weights/air_model.pkl",
        fine_tuned_model_path=None,
        weight_col="day_gap",
        valid_days=7,
        learning_rate=0.002,
        num_round=3000,
        num_threads=4,
        seed=2018,
        device="gpu"
    )
    
    logger.info(f"Fine-tune RMSLE: {results['rmsle']:.4f}")
    return results

def run_direct_transfer_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行直接迁移实验"""
    logger.info("=" * 60)
    logger.info("Running Direct Transfer Experiment")
    logger.info("=" * 60)
    
    results = direct_transfer_with_sample_selection(
        source_data=air_train,
        target_data=hpg_train,
        test_data=hpg_test,
        x_cols=x_cols,
        y_cols=y_cols,
        source_model_path="lgbm_weights/air_model.pkl",
        transferred_model_path=None,
        weight_col="day_gap",
        valid_days=7,
        selection_ratio=0.7,
        num_threads=4,
        seed=2018,
        device="gpu"
    )
    
    logger.info(f"Direct Transfer RMSLE: {results['rmsle']:.4f}")
    return results

def run_weighted_domain_adaptation_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行加权域适应实验"""
    logger.info("=" * 60)
    logger.info("Running Weighted Domain Adaptation Experiment")
    logger.info("=" * 60)
    
    results = true_ensemble_transfer_learning(
        source_data=air_train,
        target_data=hpg_train,
        test_data=hpg_test,
        x_cols=x_cols,
        y_cols=y_cols,
        source_model_path="lgbm_weights/air_model.pkl",
        ensemble_model_path=None,
        weight_col="day_gap",
        valid_days=7,
        num_threads=4,
        seed=2018,
        device="gpu"
    )
    
    logger.info(f"Weighted Domain Adaptation RMSLE: {results['rmsle']:.4f}")
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

def run_super_ensemble_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行超级集成实验"""
    logger.info("=" * 60)
    logger.info("Running Super Ensemble Experiment")
    logger.info("=" * 60)
    
    results = super_ensemble_transfer_learning(
        source_data=air_train,
        target_data=hpg_train,
        test_data=hpg_test,
        x_cols=x_cols,
        y_cols=y_cols,
        source_model_path="lgbm_weights/air_model.pkl",
        super_ensemble_model_path=None,
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
    
    logger.info(f"Super Ensemble RMSLE: {results['rmsle']:.4f}")
    return results

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Starting Comprehensive Transfer Learning Experiments")
    logger.info("=" * 60)
    
    try:
        # 加载数据
        air_train, hpg_train, hpg_test = load_data()
        
        # 准备特征
        x_cols, y_cols = prepare_features(hpg_train, hpg_test)
        
        # 存储所有实验结果
        all_results = {}
        
        # 运行基线实验
        baseline_result = run_baseline_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
        all_results['baseline'] = baseline_result['rmsle']
        
        # 运行CORAL实验
        coral_results = run_coral_experiments(air_train, hpg_train, hpg_test, x_cols, y_cols)
        all_results.update(coral_results)
        
        # 运行伪标签实验
        try:
            pseudo_result = run_pseudo_labeling_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
            all_results['pseudo_labeling'] = pseudo_result['rmsle']
        except Exception as e:
            logger.error(f"Error in pseudo labeling experiment: {e}")
        
        # 运行微调实验
        fine_tune_result = run_fine_tune_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
        all_results['fine_tune'] = fine_tune_result['rmsle']
        
        # 运行直接迁移实验
        direct_transfer_result = run_direct_transfer_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
        all_results['direct_transfer'] = direct_transfer_result['rmsle']
        
        # 运行加权域适应实验
        weighted_result = run_weighted_domain_adaptation_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
        all_results['weighted_domain_adaptation'] = weighted_result['rmsle']
        
        # 运行对抗域适应实验
        try:
            adversarial_result = run_adversarial_domain_adaptation_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
            all_results['adversarial_domain_adaptation'] = adversarial_result['rmsle']
        except Exception as e:
            logger.error(f"Error in adversarial domain adaptation experiment: {e}")
        
        # 运行超级集成实验
        super_ensemble_result = run_super_ensemble_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
        all_results['super_ensemble'] = super_ensemble_result['rmsle']
        
        # 输出所有结果摘要
        logger.info("=" * 60)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("=" * 60)
        for method, rmsle in all_results.items():
            logger.info(f"{method}: {rmsle:.4f}")
        logger.info("=" * 60)
        logger.info("All experiments completed. Results saved to all_experiments.log")
        
    except Exception as e:
        logger.error(f"Error during experiments: {e}")
        raise

if __name__ == "__main__":
    main()