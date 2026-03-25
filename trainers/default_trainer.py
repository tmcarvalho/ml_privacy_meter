"""This file contains functions for training and testing the model."""

import pdb
import time
from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, log_loss


def _log_loss_with_labels(model, y_true, y_proba):
    """Compute log loss with explicit labels to handle missing classes in a split."""
    if hasattr(model, "classes_") and len(getattr(model, "classes_", [])) == y_proba.shape[1]:
        labels = list(model.classes_)
    else:
        labels = list(range(y_proba.shape[1]))
    return log_loss(y_true, y_proba, labels=labels)


def lr_update(step: int, total_epoch: int, train_size: int, initial_lr: float) -> float:
    """
    Updates learning rate using cosine decay schedule,
    from https://github.com/tensorflow/privacy/blob/4e1fc252e4c64132ad6fcd838e93f071f38dedd7/research/mi_lira_2021/train.py#L58

    Args:
        step (int): Current step number.
        total_epoch (int): Total number of epochs.
        train_size (int): Size of the training dataset.
        initial_lr (float): Initial learning rate.

    Returns:
        float: Updated learning rate.
    """
    progress = step / (total_epoch * train_size)
    lr = initial_lr * np.cos(progress * (7 * np.pi) / (2 * 8))
    lr *= np.clip(progress * 100, 0, 1)
    return lr


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    configs: Dict,
    test_loader: torch.utils.data.DataLoader = None,
) -> torch.nn.Module:
    """
    Trains the model using the provided training data.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        configs (dict): Configuration dictionary for training.
        test_loader (torch.utils.data.DataLoader, optional): DataLoader for test data. Defaults to None.

    Returns:
        torch.nn.Module: Trained model.
    """
    # Optuna hyperparameter tuning (opt-in via configs["hyperparameter_tuning"])
    if configs.get("hyperparameter_tuning", False):
        from trainers.tuning import tune_mlp
        X_train, y_train = dataloader_to_numpy(train_loader)
        best_params = tune_mlp(X_train, y_train, model, configs)
        configs = {**configs, **best_params}

    # Ensure the model is moved to the correct device (e.g., cuda:1 or cpu)
    device = configs.get("device", "cpu")
    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, configs)

    epochs = configs.get("epochs", 1)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_update(
            step * 256, epochs, len(train_loader) * 256, 0.1
        ),
    )

    for epoch_idx in range(epochs):
        start_time = time.time()
        total_loss, correct_predictions = 0, 0

        # Set model to training mode
        model.train()

        for data, target in train_loader:
            # Ensure that both data and target are moved to the same device as the model
            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True).long(),
            )

            optimizer.zero_grad(set_to_none=True)

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_predictions += pred.eq(target).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct_predictions / len(train_loader.dataset)

        print(
            f"Epoch [{epoch_idx + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
        )

        if test_loader:
            test_loss, test_acc = inference(model, test_loader, device)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        print(f"Epoch {epoch_idx + 1} took {time.time() - start_time:.2f} seconds")

    # Unwrap DataParallel before returning so callers always get a plain module.
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.to("cpu")
    return model


def dp_train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    configs: Dict,
    test_loader: torch.utils.data.DataLoader = None,
) -> torch.nn.Module:
    """
    Trains the model using the provided training data.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        configs (dict): Configuration dictionary for training.
        test_loader (torch.utils.data.DataLoader, optional): DataLoader for test data. Defaults to None.

    Returns:
        torch.nn.Module: Trained model.
    """
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator

    # Ensure the model is moved to the correct device (e.g., cuda:1 or cpu)
    device = configs.get("device", "cpu")
    model = model.to(device)  # Make sure the model is on the correct device
    model = ModuleValidator.fix(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, configs)

    epochs = configs.get("epochs", 1)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_update(
            step * 256, epochs, len(train_loader) * 256, 0.1
        ),
    )

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
    )

    for epoch_idx in range(epochs):
        start_time = time.time()
        total_loss, correct_predictions = 0, 0

        # Set model to training mode
        model.train()

        for data, target in train_loader:
            # Ensure that both data and target are moved to the same device as the model
            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True).long(),
            )

            optimizer.zero_grad(set_to_none=True)

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_predictions += pred.eq(target).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct_predictions / len(train_loader.dataset)

        epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)

        print(
            f"Epoch [{epoch_idx + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | DP guarantee: (ε = {epsilon:.2f}, δ = {1e-5})"
        )

        if test_loader:
            test_loss, test_acc = inference(model, test_loader, device)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        print(f"Epoch {epoch_idx + 1} took {time.time() - start_time:.2f} seconds")

    # Move the model back to CPU if needed (this is optional)
    model.to("cpu")
    epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
    return model, epsilon


def train_nontorch_models(
    model,
    train_loader: torch.utils.data.DataLoader,
    configs: Dict,
    test_loader: torch.utils.data.DataLoader = None,
):
    """
    Train a model using the provided training data, supporting LightGBM and tabular foundation models.

    Args:
        model: Model object to train (e.g., LGBMClassifier)
        train_loader: PyTorch DataLoader containing the training dataset
        configs: Configuration dictionary. Must contain "train" key with model settings
        test_loader: Optional PyTorch DataLoader for testing / validation

    Returns:
        Trained model (LightGBM or other compatible models)
    """

    # Convert DataLoader to NumPy
    X_train, y_train = dataloader_to_numpy(train_loader)
    ensembling_models = ["rf"]
    boosting_models = ["lightgbm", "xgboost"]
    foundation_models = ["tabpfn", "real-tabpfn", "tabicl", "tabdpt", "tabnet", "tarte"]

    # Optuna hyperparameter tuning (opt-in via configs["hyperparameter_tuning"])
    if configs.get("hyperparameter_tuning", False) and configs["model_name"] in (ensembling_models + boosting_models + ["tabnet"]):
        from trainers.tuning import tune_nontorch
        best_params = tune_nontorch(X_train, y_train, configs["model_name"], configs)
        configs = {**configs, **best_params}

    if configs["model_name"] in boosting_models:
        model.set_params(
            n_estimators=configs.get("n_estimators", 100),
            learning_rate=configs.get("learning_rate", 0.1),
            max_depth=configs.get("max_depth", 5),
            num_leaves=configs.get("num_leaves", 31),
            min_child_samples=configs.get("min_child_samples", 20),
            subsample=configs.get("subsample", 1.0),
            colsample_bytree=configs.get("colsample_bytree", 1.0),
            reg_alpha=configs.get("reg_alpha", 0.0),
            reg_lambda=configs.get("reg_lambda", 0.0),
            random_state=configs.get("random_state", 42),
            n_jobs=-1,
            verbose=-1,
        )

    if configs["model_name"] in ensembling_models:
        model.set_params(
            n_estimators=configs.get("n_estimators", 100),
            max_depth=configs.get("max_depth", 5),
            min_samples_split=configs.get("min_samples_split", 2),
            min_samples_leaf=configs.get("min_samples_leaf", 1),
            max_features=configs.get("max_features", "sqrt"),
            random_state=configs.get("random_state", 42),
            n_jobs=-1,
        )

    if configs["model_name"] in foundation_models:
        import torch.nn.functional as F

        _old_sdpa = F.scaled_dot_product_attention

        def sdpa_ignore_gqa(*args, **kwargs):
            # Remove enable_gqa if present
            kwargs.pop("enable_gqa", None)
            return _old_sdpa(*args, **kwargs)

        F.scaled_dot_product_attention = sdpa_ignore_gqa

    if configs["model_name"] == "tabnet":
        n_train = len(X_train)
        batch_size = min(configs.get("batch_size", 256), n_train)
        virtual_batch_size = min(configs.get("virtual_batch_size", 64), batch_size)
        tabnet_params = dict(
            n_d=configs.get("n_d", 32),
            n_a=configs.get("n_d", 32),  # keep n_a == n_d
            n_steps=configs.get("n_steps", 3),
            gamma=configs.get("gamma", 1.3),
            seed=configs.get("random_state", 42),
        )
        if configs.get("tabnet_lr") is not None:
            tabnet_params["optimizer_params"] = {"lr": configs["tabnet_lr"]}
        model.set_params(**tabnet_params)
        model.fit(
            X_train, y_train,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            patience=configs.get("patience", 15),
            max_epochs=configs.get("max_epochs", 200),
        )
    else:
        model.fit(X_train, y_train)
    
    #TODO: add other models here
    y_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred)
    
    train_loss = None
    if hasattr(model, "predict_proba"): #loss requires probabilities
        train_loss = _log_loss_with_labels(model, y_train, model.predict_proba(X_train))

    print(f"Train Loss: {train_loss:.4f}" if train_loss else "")
    print(f"Train Acc: {train_acc:.4f}")

    if test_loader:
        test_loss, test_acc = inference_nontorch_models(model, test_loader)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # # Move the model back to CPU if needed (this is optional)
    # model = move_model_to_device(model)
    return model


def inference(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str
) -> Tuple[float, float]:
    """
    Evaluates the model on the provided data.

    Args:
        model (torch.nn.Module): Model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        device (str): Device to use for computation ('cpu' or 'cuda').

    Returns:
        Tuple[float, float]: Loss and accuracy on the evaluation data.
    """
    model.eval().to(device)  # Make sure the model is on the correct device
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct_predictions = 0, 0

    with torch.no_grad():
        for data, target in loader:
            # Ensure data and target are moved to the same device as the model
            data, target = data.to(device), target.to(device).long()

            output = model(data)
            loss = loss_fn(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_predictions += pred.eq(target).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / len(loader.dataset)

    return avg_loss, accuracy



def inference_nontorch_models(
    model,
    test_loader,
):
    """
    Evaluates LightGBM and tabular foundation models using a PyTorch DataLoader.

    Args:
        model: Trained model
        test_loader: torch DataLoader yielding (data, target)

    Returns:
        (loss, accuracy)
    """

    # Convert DataLoader to NumPy
    X_test, y_test = dataloader_to_numpy(test_loader)

    # Predict
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Compute loss if probabilities are available
    loss = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        loss = _log_loss_with_labels(model, y_test, y_proba)

    return loss, accuracy


def dataloader_to_numpy(loader):
    X, y = [], []
    for data, target in loader:
        d = data.cpu().numpy()
        if d.ndim == 1:
            d = d.reshape(1, -1)
        X.append(d)
        y.append(target.cpu().numpy().reshape(-1))
    return np.vstack(X), np.concatenate(y)


def get_optimizer(model: torch.nn.Module, configs: Dict) -> torch.optim.Optimizer:
    """
    Returns the optimizer based on the configuration.

    Args:
        model (torch.nn.Module): Model for which to create the optimizer.
        configs (dict): Configuration dictionary.

    Raises:
        NotImplementedError: If the specified optimizer is not supported.

    Returns:
        torch.optim.Optimizer: Initialized optimizer.
    """
    optimizer_name = configs.get("optimizer", "SGD")
    learning_rate = configs.get("learning_rate", 0.001)
    weight_decay = configs.get("weight_decay", 0.0)
    momentum = configs.get("momentum", 0.0)

    print(
        f"Using optimizer: {optimizer_name} | Learning Rate: {learning_rate} | Weight Decay: {weight_decay}"
    )

    if optimizer_name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optimizer_name == "Adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise NotImplementedError(
            f"Optimizer '{optimizer_name}' is not implemented. Choose 'SGD', 'Adam', or 'AdamW'."
        )
