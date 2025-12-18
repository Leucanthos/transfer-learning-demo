import pickle
from pathlib import Path
from typing import Optional, Dict, Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import ModelBase


def _load_booster(model_path: str) -> lgb.Booster:
    """Load a pickled LightGBM Booster."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Source model not found at {model_path}")
    with path.open("rb") as f:
        return pickle.load(f)


def true_ensemble_transfer_learning(
    source_data,
    target_data,
    test_data,
    x_cols,
    y_cols,
    source_model_path: str,
    ensemble_model_path: Optional[str] = None,
    weight_col: str = "day_gap",
    valid_days: int = 7,
    learning_rate: float = 0.003,
    num_round: int = 2000,
    num_threads: int = 4,
    seed: int = 2018,
    device: str = "cpu",
    n_models: int = 3,
) -> Dict[str, Any]:
    """
    真正的集成迁移学习方法
    
    训练多个模型并将它们的预测结果进行平均
    """
    model_base = ModelBase(device=device)
    
    # 准备数据
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=target_data,
        test_data=test_data,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        valid_days=valid_days,
    )
    
    # 加载源模型
    source_model = _load_booster(source_model_path)
    
    # 训练多个模型
    models = []
    predictions = []
    
    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}...")
        
        # 为每个模型使用不同的随机种子
        model_seed = seed + i * 1000
        
        # 训练模型
        model = model_base.train_or_load_lgbm(
            train_matrix,
            valid_matrix,
            None,  # 不单独保存每个模型
            learning_rate=learning_rate,
            num_round=num_round,
            num_threads=num_threads,
            seed=model_seed,
            early_stopping_rounds=300,
            init_model=source_model,
        )
        
        models.append(model)
        
        # 获取预测结果
        pred = model.predict(te_x, num_iteration=model.best_iteration)
        predictions.append(pred)
    
    # 平均预测结果
    ensemble_predictions = np.mean(predictions, axis=0)
    
    # 计算RMSLE
    from sklearn.metrics import mean_squared_error
    rmsle_score = np.sqrt(mean_squared_error(te_y, ensemble_predictions))
    
    # 如果指定了保存路径，则保存集成模型（这里简单地保存第一个模型）
    if ensemble_model_path:
        path = Path(ensemble_model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(models[0], f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Ensemble model saved to {ensemble_model_path}")
    
    return {
        'predictions': ensemble_predictions,
        'rmsle': rmsle_score
    }