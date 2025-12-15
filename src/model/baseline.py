import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pickle
from pathlib import Path
import pandas as pd
import warnings


def prepare_lgbm_dataset_with_weights(train_data, test_data, x_cols, y_cols, weight_col='day_gap', valid_days=7):
    """
    将原始数据转化为LightGBM可以训练的格式，并加入权重机制
    
    参数:
    train_data: 训练数据集 (DataFrame)
    test_data: 测试数据集 (DataFrame)
    x_cols: 特征列名列表
    y_cols: 目标列名列表
    weight_col: 权重列名，默认为'day_gap'
    valid_days: 用于验证的天数，默认为7天
    
    返回:
    tuple: (train_matrix, valid_matrix, te_x, te_y)
    """
    # 按visit_date排序，将最近的valid_days天作为验证集
    train_data_sorted = train_data.sort_values('visit_date')
    
    # 划分训练集和验证集
    unique_dates = sorted(train_data_sorted['visit_date'].unique())
    if len(unique_dates) <= valid_days:
        raise ValueError(f"Not enough dates for splitting. Total unique dates: {len(unique_dates)}, Valid days requested: {valid_days}")
        
    valid_cutoff_date = unique_dates[-valid_days]
    
    train_part = train_data_sorted[train_data_sorted['visit_date'] < valid_cutoff_date]
    valid_part = train_data_sorted[train_data_sorted['visit_date'] >= valid_cutoff_date]
    
    print(f"Train data date range: {train_part['visit_date'].min()} to {train_part['visit_date'].max()}")
    print(f"Valid data date range: {valid_part['visit_date'].min()} to {valid_part['visit_date'].max()}")
    print(f"Train samples: {len(train_part)}, Valid samples: {len(valid_part)}")
    
    # 准备训练数据
    tr_x = train_part[x_cols]
    tr_y = train_part[y_cols].values.ravel()
    tr_weight_raw = train_part[weight_col].values.ravel()
    
    # 准备验证数据
    val_x = valid_part[x_cols]
    val_y = valid_part[y_cols].values.ravel()
    val_weight_raw = valid_part[weight_col].values.ravel()
    
    # 准备测试数据
    te_x = test_data[x_cols]
    te_y = test_data[y_cols].values.ravel()
    
    # 计算权重：day_gap越小权重越大
    # 使用公式: weight = 1 / (1 + day_gap)，确保权重为正数
    # 对于负数day_gap，我们使用: weight = 1 / (1 + abs(day_gap)) = 1 / (1 - day_gap)
    # 添加一个小的epsilon值防止除零
    epsilon = 1e-8
    
    tr_weight = np.where(tr_weight_raw >= 0, 
                         1 / (1 + tr_weight_raw + epsilon), 
                         1 / (1 - tr_weight_raw + epsilon))
    
    val_weight = np.where(val_weight_raw >= 0, 
                          1 / (1 + val_weight_raw + epsilon), 
                          1 / (1 - val_weight_raw + epsilon))
    
    # 归一化权重
    tr_weight = tr_weight / np.mean(tr_weight)
    val_weight = val_weight / np.mean(val_weight)
    
    # 创建带权重的LightGBM数据集
    train_matrix = lgb.Dataset(tr_x, label=tr_y, weight=tr_weight)
    valid_matrix = lgb.Dataset(val_x, label=val_y, weight=val_weight)
    
    return train_matrix, valid_matrix, te_x, te_y


def train_or_load_lgbm(train_matrix, valid_matrix=None, model_path=None):
    """
    训练或加载LightGBM模型
    
    参数:
    train_matrix: LightGBM训练数据集
    valid_matrix: LightGBM验证数据集，如果为None则不使用早停
    model_path: 模型保存/加载路径 (str)，如果为None则只训练不保存
    
    返回:
    model: 训练好的LightGBM模型
    """
    # 检查是否存在已保存的模型
    if model_path and Path(model_path).exists():
        print(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    # 设置参数（参考solution.py中的参数）
    params = {
        'num_leaves': 2 ** 8 - 1,
        'objective': 'regression_l2',
        'max_depth': 9,
        'min_data_in_leaf': 50,
        'learning_rate': 0.007,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'metric': 'rmse',
        'num_threads': 4,
        'seed': 2018,
        'verbose': -1  # 静默LightGBM警告信息
    }
    
    # 训练参数
    num_round = 6000
    
    # 训练模型
    if valid_matrix is not None:
        # 使用验证集和早停
        print("Training model with early stopping...")
        model = lgb.train(
            params, 
            train_matrix, 
            num_round, 
            valid_sets=[valid_matrix],
            callbacks=[lgb.early_stopping(stopping_rounds=500), lgb.log_evaluation(period=0)]
        )
    else:
        # 不使用早停
        print("Training model without early stopping...")
        model = lgb.train(
            params, 
            train_matrix, 
            num_round,
            callbacks=[lgb.log_evaluation(period=0)]
        )
    
    # 保存模型（如果指定了路径）
    if model_path:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 使用压缩格式保存模型
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Model saved to {model_path}")
    
    return model


def evaluate_model(model, te_x, te_y):
    """
    评估模型性能
    
    参数:
    model: 训练好的模型
    te_x: 测试特征数据
    te_y: 测试标签数据
    
    返回:
    dict: 包含预测结果和RMSLE评分的字典
    """
    # 静默警告信息
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 进行预测
        test_pred = model.predict(te_x, num_iteration=model.best_iteration) if hasattr(model, 'best_iteration') else model.predict(te_x)
    
    # 计算RMSLE评分
    # 注意：输入的visitors已经是取过对数的值，所以直接计算RMSE即可
    rmsle_score = np.sqrt(mean_squared_error(te_y, test_pred))
    
    return {
        'predictions': test_pred,
        'rmsle': rmsle_score
    }


def rmsle(y_true, y_pred):
    """
    计算RMSLE指标
    RMSLE = sqrt(1/n * sum((log(p_i + 1) - log(a_i + 1))^2))
    
    参数:
    y_true: 真实值数组
    y_pred: 预测值数组
    
    返回:
    rmsle_score: RMSLE评分
    """
    # 注意：输入的visitors已经是取过对数的值
    # 所以我们直接计算RMSE即可
    return np.sqrt(mean_squared_error(y_true, y_pred))