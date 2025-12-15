import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Iterable

import lightgbm as lgb
import numpy as np

from .base import ModelBase


def _load_booster(model_path: str) -> lgb.Booster:
    """Load a pickled LightGBM Booster."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Source model not found at {model_path}")
    with path.open("rb") as f:
        return pickle.load(f)


def finetune_from_source(
    train_data,
    test_data,
    x_cols,
    y_cols,
    source_model_path: str,
    finetuned_model_path: Optional[str] = None,
    weight_col: str = "day_gap",
    valid_days: int = 7,
    learning_rate: float = 0.003,
    num_round: int = 2000,
    num_threads: int = 4,
    seed: int = 2018,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Fine-tune a pre-trained LightGBM model from source domain on target-domain data.

    The source model is loaded from ``source_model_path`` and used as ``init_model``.
    Target data is split by date into train/valid using ``valid_days``.
    """
    model_base = ModelBase(device=device)
    
    train_matrix, valid_matrix, te_x, te_y = model_base.prepare_lgbm_dataset_with_weights(
        train_data=train_data,
        test_data=test_data,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        valid_days=valid_days,
    )

    source_model = _load_booster(source_model_path)

    finetuned = model_base.train_or_load_lgbm(
        train_matrix,
        valid_matrix,
        finetuned_model_path,
        learning_rate=learning_rate,
        num_round=num_round,
        num_threads=num_threads,
        seed=seed,
        early_stopping_rounds=300,
        init_model=source_model,
    )

    return model_base.evaluate_model(finetuned, te_x, te_y)


def _covariance(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute covariance with small diagonal jitter to ensure PSD."""
    c = np.cov(mat, rowvar=False)
    c += np.eye(c.shape[0]) * eps
    return c


def _whiten_and_color(source: np.ndarray, c_s: np.ndarray, c_t: np.ndarray) -> np.ndarray:
    """CORAL transform: whiten source then re-color with target covariance."""
    # eigendecomposition for symmetric matrices
    d_s, e_s = np.linalg.eigh(c_s)
    d_t, e_t = np.linalg.eigh(c_t)

    # build whitening and coloring matrices
    w_s = e_s @ np.diag(1.0 / np.sqrt(np.maximum(d_s, 1e-12))) @ e_s.T
    c_t_half = e_t @ np.diag(np.sqrt(np.maximum(d_t, 1e-12))) @ e_t.T

    return (source @ w_s) @ c_t_half


def coral_align_source_to_target(
    source_df,
    target_df,
    feature_cols: Iterable[str],
    eps: float = 1e-6,
):
    """
    Align source features to target domain using CORAL (Correlation Alignment).

    Only feature columns are transformed; other columns (e.g., labels, date, weights)
    are kept as-is. Returns a transformed copy of the source dataframe.
    """
    src = source_df.copy()
    tgt = target_df.copy()

    # Extract features
    src_feat = src[list(feature_cols)].to_numpy(dtype=float)
    tgt_feat = tgt[list(feature_cols)].to_numpy(dtype=float)

    # Center
    src_mean = src_feat.mean(axis=0, keepdims=True)
    tgt_mean = tgt_feat.mean(axis=0, keepdims=True)
    src_centered = src_feat - src_mean

    # Covariance
    c_s = _covariance(src_centered, eps)
    c_t = _covariance(tgt_feat - tgt_mean, eps)

    # CORAL transform
    src_aligned = _whiten_and_color(src_centered, c_s, c_t) + tgt_mean

    # Write back
    src[list(feature_cols)] = src_aligned
    return src

