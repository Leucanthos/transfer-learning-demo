import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pickle
from pathlib import Path
import pandas as pd
import warnings
from contextlib import contextmanager


class ModelBase:
    """
    模型基类，提供通用的模型训练、保存、加载和评估功能
    支持设备选择（CPU/GPU）
    """
    
    def __init__(self, device="cpu"):
        """
        初始化模型基类
        
        参数:
        device: 设备类型，"cpu" 或 "gpu"
        """
        self.device = device.lower()
        if self.device not in ["cpu", "gpu"]:
            raise ValueError("Device must be either 'cpu' or 'gpu'")
            
    @contextmanager
    def device_context(self, device=None):
        """
        设备上下文管理器，用于临时切换设备
        
        参数:
        device: 设备类型，"cpu" 或 "gpu"，如果为None则使用实例默认设备
        """
        old_device = self.device
        if device is not None:
            self.device = device.lower()
        try:
            yield self
        finally:
            self.device = old_device
            
    def _get_params(self, **kwargs):
        """
        获取模型参数，根据设备类型添加相应参数
        """
        params = {
            'num_leaves': kwargs.get('num_leaves', 2 ** 8 - 1),
            'objective': kwargs.get('objective', 'regression_l2'),
            'max_depth': kwargs.get('max_depth', 9),
            'min_data_in_leaf': kwargs.get('min_data_in_leaf', 50),
            'learning_rate': kwargs.get('learning_rate', 0.007),
            'feature_fraction': kwargs.get('feature_fraction', 0.6),
            'bagging_fraction': kwargs.get('bagging_fraction', 0.8),
            'bagging_freq': kwargs.get('bagging_freq', 1),
            'metric': kwargs.get('metric', 'rmse'),
            'num_threads': kwargs.get('num_threads', 4),
            'seed': kwargs.get('seed', 2018),
            'verbose': kwargs.get('verbose', -1)  # 静默LightGBM警告信息
        }
        
        # 根据设备类型添加设备相关参数
        if self.device == "gpu":
            params['device'] = 'gpu'
            params['gpu_platform_id'] = kwargs.get('gpu_platform_id', 0)
            params['gpu_device_id'] = kwargs.get('gpu_device_id', 0)
            
        return params
    
    def prepare_lgbm_dataset_with_weights(self, train_data, test_data, x_cols, y_cols, weight_col='day_gap', valid_days=7):
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

    def train_or_load_lgbm(self, train_matrix, valid_matrix=None, model_path=None, **kwargs):
        """
        训练或加载LightGBM模型
        
        参数:
        train_matrix: LightGBM训练数据集
        valid_matrix: LightGBM验证数据集，如果为None则不使用早停
        model_path: 模型保存/加载路径 (str)，如果为None则只训练不保存
        **kwargs: 其他参数，如num_round, early_stopping_rounds等
        
        返回:
        model: 训练好的LightGBM模型
        """
        # 检查是否存在已保存的模型
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        
        # 获取参数
        params = self._get_params(**kwargs)
        
        # 训练参数
        num_round = kwargs.get('num_round', 6000)
        early_stopping_rounds = kwargs.get('early_stopping_rounds', 500)
        
        # 训练模型
        if valid_matrix is not None:
            # 使用验证集和早停
            print(f"Training model with early stopping on {self.device.upper()}...")
            model = lgb.train(
                params, 
                train_matrix, 
                num_round, 
                valid_sets=[valid_matrix],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds), 
                    lgb.log_evaluation(period=0)
                ]
            )
        else:
            # 不使用早停
            print(f"Training model without early stopping on {self.device.upper()}...")
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

    def evaluate_model(self, model, te_x, te_y):
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

    @staticmethod
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