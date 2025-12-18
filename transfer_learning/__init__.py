"""
迁移学习模块初始化文件

该模块实现了多种迁移学习方法，按照技术类别组织：
1. 基础方法 (base.py, baseline.py)
2. 单一策略方法 (transfer.py)
3. 集成方法 (ensemble_transfer.py)
4. 高级方法 (advanced_transfer.py)
5. 综合方法 (super_ensemble.py)
"""

from .base import ModelBase
from .baseline import train_or_load_lgbm, evaluate_model

# 单一策略迁移学习方法
from .transfer import (
    pseudo_label_transfer_learning,
    fine_tune_transfer_learning,
    direct_transfer_with_sample_selection
)

# 集成迁移学习方法
from .ensemble_transfer import true_ensemble_transfer_learning

# 高级迁移学习方法
from .advanced_transfer import adversarial_domain_adaptation

# 综合迁移学习方法
from .super_ensemble import super_ensemble_transfer_learning

__all__ = [
    # 基础类和函数
    "ModelBase",
    "train_or_load_lgbm",
    "evaluate_model",
    
    # 单一策略方法
    "pseudo_label_transfer_learning",
    "fine_tune_transfer_learning",
    "direct_transfer_with_sample_selection",
    
    # 集成方法
    "true_ensemble_transfer_learning",
    
    # 高级方法
    "adversarial_domain_adaptation",
    
    # 综合方法
    "super_ensemble_transfer_learning",
]