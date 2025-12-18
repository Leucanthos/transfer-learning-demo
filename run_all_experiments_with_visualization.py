import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import date, timedelta

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.transfer import pseudo_label_transfer_learning, fine_tune_transfer_learning
from src.model.advanced_transfer import adversarial_domain_adaptation
from src.model.super_ensemble import super_ensemble_transfer_learning
from src.model.baseline import evaluate_model
from src.model.base import ModelBase

# 设置随机种子
np.random.seed(2018)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("all_experiments_with_visualization.log"),
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
    plt.savefig(f'visualization_{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_feature_importance(model, feature_names, title, filename):
    """可视化特征重要性"""
    if hasattr(model, 'feature_importance'):
        importance = model.feature_importance()
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance_df, x='importance', y='feature')
        plt.title(f'{title} - Top 20 Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()

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
        
        # 可视化结果
        visualize_predictions_vs_actual(results, "Pseudo Labeling", "pseudo_labeling")
        # 可视化特征重要性
        if 'model' in results:
            visualize_feature_importance(results['model'], expected_feature_names, 
                                      "Pseudo Labeling", "pseudo_labeling")
        
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
        
        # 可视化结果
        visualize_predictions_vs_actual(results, "Adversarial Domain Adaptation", "adversarial_da")
        
        return results
    except Exception as e:
        logger.error(f"Error in adversarial domain adaptation experiment: {e}")
        return None

def run_fine_tune_experiments(air_train, hpg_train, hpg_test, x_cols, y_cols):
    """运行不同参数的微调实验"""
    logger.info("=" * 60)
    logger.info("Running Fine-tune Experiments with Different Parameters")
    logger.info("=" * 60)
    
    results = {}
    
    # 不同的学习率和迭代次数组合
    params_combinations = [
        {'lr': 0.001, 'rounds': 1000},
        {'lr': 0.01, 'rounds': 1000},
        {'lr': 0.05, 'rounds': 1000},
    ]
    
    best_rmsle = float('inf')
    best_result = None
    
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
                fine_tune_lr=lr,
                weight_col="day_gap",
                valid_days=7,
                device="gpu",
                seed=2018,
                num_round=rounds
            )
            results[exp_name] = result['rmsle']
            logger.info(f"Fine-tune with learning rate {lr}, rounds {rounds} - RMSLE: {result['rmsle']:.4f}")
            
            # 保存最佳结果用于可视化
            if result['rmsle'] < best_rmsle:
                best_rmsle = result['rmsle']
                best_result = result
                
        except Exception as e:
            logger.error(f"Error in fine-tune experiment with lr={lr}, rounds={rounds}: {e}")
    
    # 可视化最佳结果
    if best_result:
        visualize_predictions_vs_actual(best_result, "Fine-tune (Best)", "fine_tune_best")
    
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
        {'topk': 30, 'lr_phase1': 0.01, 'lr_phase2': 0.002},
    ]
    
    best_rmsle = float('inf')
    best_result = None
    
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
                weight_col="day_gap",
                valid_days=7,
                learning_rate_phase1=lr_phase1,
                learning_rate_phase2=lr_phase2,
                num_round_phase1=500,
                num_round_phase2=1000,
                device="gpu",
                seed=2018
            )
            results[exp_name] = result['rmsle']
            logger.info(f"Super Ensemble with topk {topk}, learning rate phase1 {lr_phase1}, phase2 {lr_phase2} - RMSLE: {result['rmsle']:.4f}")
            
            # 保存最佳结果用于可视化
            if result['rmsle'] < best_rmsle:
                best_rmsle = result['rmsle']
                best_result = result
                
        except Exception as e:
            logger.error(f"Error in super ensemble experiment with topk={topk}, lr_phase1={lr_phase1}, lr_phase2={lr_phase2}: {e}")
    
    # 可视化最佳结果
    if best_result:
        visualize_predictions_vs_actual(best_result, "Super Ensemble (Best)", "super_ensemble_best")
    
    return results

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
    plt.savefig('experiment_results_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Starting All Transfer Learning Experiments with Visualization")
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
            all_results['Pseudo Labeling'] = pseudo_result['rmsle']
        
        # 运行对抗域适应实验
        logger.info("Starting Adversarial Domain Adaptation Experiment")
        adversarial_result = run_adversarial_domain_adaptation_experiment(air_train, hpg_train, hpg_test, x_cols, y_cols)
        if adversarial_result:
            all_results['Adversarial DA'] = adversarial_result['rmsle']
        
        # 运行微调实验
        logger.info("Starting Fine-tune Experiments")
        finetune_results = run_fine_tune_experiments(air_train, hpg_train, hpg_test, x_cols, y_cols)
        # 只保存最好的微调结果
        if finetune_results:
            best_finetune = min(finetune_results.items(), key=lambda x: x[1])
            all_results[f'Fine-tune ({best_finetune[0]})'] = best_finetune[1]
        
        # 运行超级集成微调实验
        logger.info("Starting Super Ensemble Fine-tune Experiments")
        super_ensemble_results = run_super_ensemble_finetune_experiments(air_train, hpg_train, hpg_test, x_cols, y_cols)
        # 只保存最好的超级集成结果
        if super_ensemble_results:
            best_super_ensemble = min(super_ensemble_results.items(), key=lambda x: x[1])
            all_results[f'Super Ensemble ({best_super_ensemble[0]})'] = best_super_ensemble[1]
        
        # 输出所有结果摘要
        logger.info("=" * 60)
        logger.info("ALL EXPERIMENTS RESULTS SUMMARY")
        logger.info("=" * 60)
        for method, rmsle in sorted(all_results.items(), key=lambda item: item[1]):
            logger.info(f"{method}: {rmsle:.4f}")
        logger.info("=" * 60)
        
        # 绘制结果对比图
        plot_results_comparison(all_results)
        
        logger.info("All experiments completed. Results saved to all_experiments_with_visualization.log")
        logger.info("Visualizations saved as PNG files.")
        
    except Exception as e:
        logger.error(f"Error during experiments: {e}")
        raise

if __name__ == "__main__":
    main()