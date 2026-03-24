"""Optuna-based hyperparameter tuning for RF, LightGBM, TabNet, and MLP."""

import json
import os
import numpy as np
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _best_params_path(configs):
    """Return path for saving/loading best hyperparameters."""
    log_dir = configs.get("log_dir", "ml_privacy_meter/logs")
    return os.path.join(log_dir, "best_params.json")


def _save_best_params(params, configs):
    path = _best_params_path(configs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"[Optuna] Best params saved to {path}")


def _load_best_params(configs):
    path = _best_params_path(configs)
    if os.path.exists(path):
        with open(path) as f:
            params = json.load(f)
        print(f"[Optuna] Loaded saved best params from {path}: {params}")
        return params
    return None


def tune_nontorch(X_train, y_train, model_name, configs):
    """
    Tune hyperparameters for RF, LightGBM, or TabNet using Optuna TPE.
    Saves best params to log_dir/best_params.json and reuses them on subsequent runs.

    Args:
        X_train: numpy array of training features
        y_train: numpy array of training labels
        model_name: one of 'rf', 'lightgbm', 'tabnet'
        configs: training config dict (read for n_trials, cv, random_state, log_dir)

    Returns:
        dict of best hyperparameters (keys match configs keys used in set_params)
    """
    saved = _load_best_params(configs)
    if saved is not None:
        return saved

    n_trials = configs.get("tuning_n_trials", 30)
    cv = configs.get("tuning_cv", 3)
    seed = configs.get("random_state", 42)

    print(f"\n[Optuna] Tuning {model_name} | {n_trials} trials | {cv}-fold CV")

    if model_name == "rf":
        best = _tune_rf(X_train, y_train, n_trials, cv, seed)
    elif model_name == "lightgbm":
        best = _tune_lightgbm(X_train, y_train, n_trials, cv, seed)
    elif model_name == "tabnet":
        best = _tune_tabnet(X_train, y_train, configs, n_trials, cv, seed)
    else:
        print(f"[Optuna] No tuning implemented for {model_name}, skipping.")
        return {}

    _save_best_params(best, configs)
    return best


def _tune_rf(X_train, y_train, n_trials, cv, seed):
    from sklearn.ensemble import RandomForestClassifier

    n_samples, n_features = X_train.shape

    # max_depth: scale with log2(n_features) so shallow datasets don't over-search.
    # min 3 to allow some depth; cap at 30 for high-dimensional data.
    max_depth_cap = max(3, min(30, int(np.ceil(np.log2(n_features + 1)) * 3)))
    max_depth_choices = [d for d in [None, 3, 5, 8, 10, 15, 20, max_depth_cap] if d is None or d <= max_depth_cap]
    max_depth_choices = list(dict.fromkeys(max_depth_choices))  # deduplicate, preserve order

    # min_samples_leaf: scale with dataset size so large datasets can regularize more.
    max_leaf = max(10, n_samples // 200)

    # max_features: use float fraction so it scales with any feature count.
    # sqrt(n) / n and log2(n) / n shrink fast — floor at 0.05 for high-dim datasets.
    max_features_low = max(0.05, round(np.log2(n_features + 1) / n_features, 3))
    max_features_high = min(1.0, max(0.5, round(np.sqrt(n_features) / n_features * 2, 3)))

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_categorical("max_depth", max_depth_choices),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, max_leaf),
            "max_features": trial.suggest_float("max_features", max_features_low, max_features_high),
            "random_state": seed,
            "n_jobs": -1,
        }
        scores = [
            accuracy_score(
                y_train[val_idx],
                RandomForestClassifier(**params).fit(X_train[train_idx], y_train[train_idx]).predict(X_train[val_idx])
            )
            for train_idx, val_idx in kf.split(X_train, y_train)
        ]
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"[Optuna] Best RF params: {best} | CV acc: {study.best_value:.4f}")
    print(f"  (search space: max_depth={max_depth_choices}, max_features=[{max_features_low:.3f},{max_features_high:.3f}], min_samples_leaf=[1,{max_leaf}])")
    return best


def _tune_lightgbm(X_train, y_train, n_trials, cv, seed):
    from lightgbm import LGBMClassifier

    n_samples, n_features = X_train.shape

    # num_leaves: more features -> more possible interactions; cap at 255 (LightGBM hard limit).
    max_leaves = min(255, max(31, n_features * 3))

    # min_child_samples: scale with dataset size for appropriate regularization.
    min_child_low = max(5, n_samples // 500)
    min_child_high = max(50, n_samples // 50)

    # colsample_bytree: for low-dim data allow full feature use; high-dim benefits from subsampling.
    colsample_low = max(0.3, min(0.8, round(np.sqrt(n_features) / n_features * 3, 2)))

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, max_leaves),
            "min_child_samples": trial.suggest_int("min_child_samples", min_child_low, min_child_high),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", colsample_low, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        scores = [
            accuracy_score(
                y_train[val_idx],
                LGBMClassifier(**params).fit(X_train[train_idx], y_train[train_idx]).predict(X_train[val_idx])
            )
            for train_idx, val_idx in kf.split(X_train, y_train)
        ]
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"[Optuna] Best LightGBM params: {best} | CV acc: {study.best_value:.4f}")
    print(f"  (search space: num_leaves=[15,{max_leaves}], min_child_samples=[{min_child_low},{min_child_high}], colsample_bytree=[{colsample_low:.2f},1.0])")
    return best


def _tune_tabnet(X_train, y_train, configs, n_trials, cv, seed):
    from pytorch_tabnet.tab_model import TabNetClassifier

    n_samples, n_features = X_train.shape

    # n_d: embedding width — cap relative to feature count to avoid over-parameterizing low-dim data.
    max_n_d = min(64, max(8, n_features * 4))
    n_d_choices = [d for d in [8, 16, 32, 64] if d <= max_n_d] or [8]

    # batch_size: exclude sizes larger than half the training set.
    valid_batch_sizes = [b for b in [128, 256, 512, 1024] if b <= n_samples // 2] or [min(128, n_samples // 2)]

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    def objective(trial):
        n_d = trial.suggest_categorical("n_d", n_d_choices)
        n_steps = trial.suggest_int("n_steps", 3, 8)
        gamma = trial.suggest_float("gamma", 1.0, 2.0)
        lr = trial.suggest_float("tabnet_lr", 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", valid_batch_sizes)

        scores = []
        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr = X_train[train_idx].astype(np.float32)
            X_val = X_train[val_idx].astype(np.float32)
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            bs = min(batch_size, len(X_tr))
            vbs = min(bs // 4, bs)

            m = TabNetClassifier(
                n_d=n_d, n_a=n_d,
                n_steps=n_steps,
                gamma=gamma,
                optimizer_params={"lr": lr},
                seed=seed,
                verbose=0,
            )
            m.fit(X_tr, y_tr, batch_size=bs, virtual_batch_size=vbs, patience=5, max_epochs=50)
            scores.append(accuracy_score(y_val, m.predict(X_val)))

        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"[Optuna] Best TabNet params: {best} | CV acc: {study.best_value:.4f}")
    print(f"  (search space: n_d={n_d_choices}, batch_size={valid_batch_sizes})")
    return best


def tune_mlp(X_train, y_train, model, base_configs):
    """
    Tune MLP hyperparameters using Optuna with k-fold CV.

    Args:
        X_train: numpy array of training features
        y_train: numpy array of training labels
        model: existing MLP instance (used to extract architecture)
        base_configs: training config dict

    Returns:
        dict of best hyperparameters
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from models.mlp import MLP
    from trainers.default_trainer import train, inference

    saved = _load_best_params(base_configs)
    if saved is not None:
        return saved

    n_trials = base_configs.get("tuning_n_trials", 30)
    cv = base_configs.get("tuning_cv", 3)
    seed = base_configs.get("random_state", 42)
    tune_epochs = max(10, base_configs.get("epochs", 100) // 5)

    # Extract architecture from existing model
    in_shape = model.feature_extractor[0].in_features
    num_classes = model.classifier.out_features

    n_samples, _ = X_train.shape

    # batch_size: exclude sizes larger than half the training fold.
    valid_batch_sizes = [b for b in [128, 256, 512] if b <= n_samples // 2] or [min(128, n_samples // 2)]

    print(f"\n[Optuna] Tuning MLP | {n_trials} trials | {cv}-fold CV | {tune_epochs} epochs/trial")

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        bs = trial.suggest_categorical("batch_size", valid_batch_sizes)
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])

        scores = []
        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            train_dl = DataLoader(
                TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
                batch_size=bs, shuffle=True,
            )
            val_dl = DataLoader(
                TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
                batch_size=bs,
            )

            m = MLP(in_shape=in_shape, num_classes=num_classes)
            tune_configs = {
                **base_configs,
                "learning_rate": lr,
                "weight_decay": wd,
                "batch_size": bs,
                "optimizer": optimizer,
                "epochs": tune_epochs,
                "hyperparameter_tuning": False,  # prevent recursion
            }
            m = train(m, train_dl, tune_configs)
            _, val_acc = inference(m, val_dl, base_configs.get("device", "cpu"))
            scores.append(val_acc)

        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"[Optuna] Best MLP params: {best} | CV acc: {study.best_value:.4f}")
    _save_best_params(best, base_configs)
    return best
