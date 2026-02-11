import os.path
from typing import Optional, Union

import numpy as np
from sklearn.metrics import log_loss
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, AutoTokenizer

from dataset.utils import load_dataset_subsets


def move_model_to_device(model, device):
    if hasattr(model, "to"):
        return model.to(device)
    return model


def get_softmax(
    model: Union[PreTrainedModel, torch.nn.Module],
    samples: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: str,
    temp: float = 1.0, # normal softmax
    pad_token_id: Optional[int] = None,
) -> np.ndarray:
    """
    Get the model's softmax probabilities for the given inputs and expected outputs.

    Args:
        model (PreTrainedModel or torch.nn.Module): Model instance.
        samples (torch.Tensor): Model input.
        labels (torch.Tensor): Model expected output.
        batch_size (int): Batch size for getting signals.
        device (str): Device used for computing signals.
        temp (float): Temperature used in softmax computation.
        pad_token_id (Optional[int]): Padding token ID to ignore in aggregation.

    Returns:
        all_softmax_list (np.array): softmax value of all samples
    """

    model = move_model_to_device(model, device)
    model.eval()
    with torch.no_grad():
        softmax_list = []
        batched_samples = torch.split(samples, batch_size) # Splits inputs and labels into mini-batches
        batched_labels = torch.split(labels, batch_size)

        for x, y in tqdm(
            zip(batched_samples, batched_labels),
            total=len(batched_samples),
            desc="Computing softmax",
        ):
            x = x.to(device)
            y = y.to(device)

            pred = model(x) # forward pass to get logits
            if isinstance(model, PreTrainedModel):
                logits = pred.logits
                logit_signals = torch.div(logits, temp)
                log_probs = torch.log_softmax(logit_signals, dim=-1)
                true_class_log_probs = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)
                # Mask out padding tokens
                mask = (
                    y != pad_token_id
                    if pad_token_id is not None
                    else torch.ones_like(y, dtype=torch.bool)
                )
                true_class_log_probs = true_class_log_probs * mask
                sequence_probs = torch.exp(
                    (true_class_log_probs * mask).sum(1) / mask.sum(1)
                )
                softmax_list.append(sequence_probs.to("cpu").view(-1, 1))
            else:
                logit_signals = torch.div(pred, temp) # apply temperature scaling
                max_logit_signals, _ = torch.max(logit_signals, dim=1) # find max per sample
                # This is to avoid overflow when exp(logit_signals)
                logit_signals = torch.sub(
                    logit_signals, max_logit_signals.reshape(-1, 1)
                )
                exp_logit_signals = torch.exp(logit_signals) # manual softmax computation
                exp_logit_sum = exp_logit_signals.sum(dim=1).reshape(-1, 1)
                true_exp_logit = exp_logit_signals.gather(1, y.reshape(-1, 1))
                softmax_list.append(torch.div(true_exp_logit, exp_logit_sum).to("cpu")) # final softmax prob -- returns probabilities for TRUE labels only
        all_softmax_list = np.concatenate(softmax_list)
    model.to("cpu")
    return all_softmax_list


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def get_probs_nontorch_models(
    model,
    logger,
    samples,
    labels,
    batch_size,
    temp: float = 1.0,
) -> np.ndarray:
    """
    Compute the softmax probability assigned to the true class
    for sklearn-style models.

    Returns:
        np.ndarray of shape (n_samples, 1)
    """

    softmax_list = []
    model_exposing = None
    for start in range(0, len(samples), batch_size):
        end = start + batch_size
        x_batch = samples[start:end]
        y_batch = labels[start:end]

        # get logits
        if hasattr(model, "predict_logits"):
            logits = model.predict_logits(x_batch)
            model_exposing = "predict_logits"

        elif hasattr(model, "decision_function"):
            logits = model.decision_function(x_batch)

            # Binary sklearn case → convert to 2-class logits
            if logits.ndim == 1:
                logits = np.stack([-logits, logits], axis=1)
            model_exposing = "decision_function"

        elif hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_batch)
            logits = np.log(probs + 1e-12)  # pseudo-logits=\
            model_exposing = "predict_proba"

        else:
            raise ValueError("Model must expose logits or probabilities.")

        # temperature scaling
        logits = logits / temp

        # softmax
        probs = stable_softmax(logits)

        # select true class probability
        true_probs = probs[np.arange(len(y_batch)), y_batch]
        softmax_list.append(true_probs.reshape(-1, 1))

    logger.info("Model exposing via {}.".format(model_exposing))

    return np.concatenate(softmax_list)


def get_loss(
    model: Union[PreTrainedModel, torch.nn.Module],
    samples: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: str,
    pad_token_id: Optional[int] = None,
) -> np.ndarray:
    """
    Get the model's loss for the given inputs and expected outputs.

    Args:
        model (PreTrainedModel or torch.nn.Module): Model instance.
        samples (torch.Tensor): Model input.
        labels (torch.Tensor): Model expected output.
        batch_size (int): Batch size for getting signals.
        device (str): Device used for computing signals.
        pad_token_id (Optional[int]): Padding token ID to ignore in aggregation.

    Returns:
        all_loss_list (np.array): Loss value of all samples
    """
    model = move_model_to_device(model, device)
    model.eval()
    with torch.no_grad():
        loss_list = []
        batched_samples = torch.split(samples, batch_size)
        batched_labels = torch.split(labels, batch_size)
        for x, y in zip(batched_samples, batched_labels):
            x = x.to(device)
            y = y.to(device)
            if isinstance(model, PreTrainedModel):
                logit_signals = model(x).logits
                loss = torch.nn.CrossEntropyLoss(
                    reduction="none", ignore_index=pad_token_id
                )(logit_signals.transpose(1, 2), y)
                mask = loss != 0
                loss = (loss * mask).sum(1) / mask.sum(1)
                loss_list.append(loss.cpu().detach().numpy().reshape(batch_size, -1))
            else:
                logit_signals = model(x)
                loss_list.append(
                    F.cross_entropy(logit_signals, y.ravel(), reduction="none").to(
                        "cpu"
                    )
                )
        all_loss_list = np.concatenate(loss_list).reshape((-1, 1))
    model.to("cpu")
    return all_loss_list


def get_model_signals(models_list, dataset, configs, logger, is_population=False):
    """Function to get models' signals (softmax, loss, logits) on a given dataset.

    Args:
        models_list (list): List of models for computing (softmax, loss, logits) signals from them.
        dataset (torchvision.datasets): The whole dataset.
        configs (dict): Configurations of the tool.
        logger (logging.Logger): Logger object for the current run.
        is_population (bool): Whether the signals are computed on population data.

    Returns:
        signals (np.array): Signal value for all samples in all models
    """
    new_models = ["lightgbm", "rf", "tabpfn", "tabicl", "tabdpt", "tarte"] #TODO: add more
    # Check if signals are available on disk
    signal_file_name = (
        f"{configs['audit']['algorithm'].lower()}_ramia_signals"
        if configs.get("ramia", None)
        else f"{configs['audit']['algorithm'].lower()}_signals"
    )
    signal_file_name += "_pop.npy" if is_population else ".npy"
    signal_dir = os.path.join(configs['run']['log_dir'], 'signals')
    signal_path = os.path.join(signal_dir, signal_file_name)
    if os.path.exists(signal_path):
        signals = np.load(signal_path)
        if configs.get("ramia", None) is None:
            expected_size = len(dataset)
            signal_source = "training data size"
        else:
            expected_size = len(dataset) * configs["ramia"]["sample_size"]
            signal_source = f"training data size multiplied by ramia sample size ({configs['ramia']['sample_size']})"

        if signals.shape[0] == expected_size:
            logger.info("Signals loaded from disk successfully.")
            return signals
        else:
            logger.warning(
                f"Signals shape ({signals.shape[0]}) does not match the expected size ({expected_size}). "
                f"This mismatch is likely due to a change in the {signal_source}."
            )
            logger.info("Ignoring the signals on disk and recomputing.")

    batch_size = configs["audit"]["batch_size"]  # Batch size used for inferring signals
    model_name = configs["train"]["model_name"]  # Algorithm used for training models
    device = configs["audit"]["device"]  # GPU device used for inferring signals
    if "tokenizer" in configs["data"].keys():
        tokenizer = AutoTokenizer.from_pretrained(
            configs["data"]["tokenizer"], clean_up_tokenization_spaces=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = None

    dataset_samples = np.arange(len(dataset))
    data, targets = load_dataset_subsets(
        dataset, dataset_samples, model_name, batch_size, device
    )

    signals = []
    logger.info("Computing signals for all models.")
    if configs.get("ramia", None) and not is_population:
        if len(data.shape) != 2:
            data = data.view(-1, *data.shape[2:])
            targets = targets.view(data.shape[0], -1)
    for model in models_list:
        if model_name.lower() in new_models:
            # Use the LightGBM and foundation models
            signals.append(
                get_probs_nontorch_models(model, logger, data, targets, batch_size)
            )
        else:
            # Use the original PyTorch/HuggingFace function
            signals.append(
                get_softmax(model, data, targets, batch_size, device, pad_token_id=pad_token_id)
            )

    signals = np.concatenate(signals, axis=1)
    os.makedirs(signal_dir, exist_ok=True)
    np.save(signal_path, signals)
    logger.info("Signals saved to disk.")
    return signals
