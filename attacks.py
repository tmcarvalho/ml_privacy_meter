from typing import Any

import numpy as np
from sklearn.metrics import auc, roc_curve


def _get_reference_columns(
    all_signals: np.ndarray,
    target_model_idx: int,
    num_reference_models: int,
) -> list:
    """Return reference model column indices (excluding target and its paired model)."""
    paired_model_idx = (
        target_model_idx + 1 if target_model_idx % 2 == 0 else target_model_idx - 1
    )
    return [
        i
        for i in range(all_signals.shape[1])
        if i != target_model_idx and i != paired_model_idx
    ][: 2 * num_reference_models]


def get_rmia_out_signals(
    all_signals: np.ndarray,
    all_memberships: np.ndarray,
    target_model_idx: int,
    num_reference_models: int,
) -> np.ndarray:
    """
    Get average prediction probability of samples over OUT reference models (excluding the target model).

    Args:
        all_signals (np.ndarray): Softmax value of all samples in every model.
        all_memberships (np.ndarray): Membership matrix for all models (if a sample is used for training a model).
        target_model_idx (int): Target model index.
        num_reference_models (int): Number of reference models used for the attack.

    Returns:
        np.ndarray: Average softmax value for each sample over OUT reference models.
    """
    columns = _get_reference_columns(all_signals, target_model_idx, num_reference_models)
    selected_signals = all_signals[:, columns]
    non_members = ~all_memberships[:, columns]
    out_signals = selected_signals * non_members
    # Sort so that only the non-zero (out) signals are kept
    out_signals = -np.sort(-out_signals, axis=1)[:, :num_reference_models]
    return out_signals


def get_rmia_in_signals(
    all_signals: np.ndarray,
    all_memberships: np.ndarray,
    target_model_idx: int,
    num_reference_models: int,
) -> np.ndarray:
    """
    Get average prediction probability of samples over IN reference models (online variant).

    Args:
        all_signals (np.ndarray): Softmax value of all samples in every model.
        all_memberships (np.ndarray): Membership matrix for all models.
        target_model_idx (int): Target model index.
        num_reference_models (int): Number of reference models used for the attack.

    Returns:
        np.ndarray: Average softmax value for each sample over IN reference models.
    """
    columns = _get_reference_columns(all_signals, target_model_idx, num_reference_models)
    selected_signals = all_signals[:, columns]
    members = all_memberships[:, columns]
    in_signals = selected_signals * members
    # Sort so that only the non-zero (in) signals are kept
    in_signals = -np.sort(-in_signals, axis=1)[:, :num_reference_models]
    return in_signals


def tune_offline_a(
    target_model_idx: int,
    all_signals: np.ndarray,
    population_signals: np.ndarray,
    all_memberships: np.ndarray,
    logger: Any,
) -> (float, np.ndarray, np.ndarray):
    """
    Fine-tune coefficient offline_a used in RMIA.

    Args:
        target_model_idx (int): Index of the target model.
        all_signals (np.ndarray): Softmax value of all samples in two models (target and reference).
        population_signals (np.ndarray): Population signals.
        all_memberships (np.ndarray): Membership matrix for all models.
        logger (Any): Logger object for the current run.

    Returns:
        float: Optimized offline_a obtained by attacking a paired model with the help of the reference models.
    """
    paired_model_idx = (
        target_model_idx + 1 if target_model_idx % 2 == 0 else target_model_idx - 1
    )
    logger.info(f"Fine-tuning offline_a using paired model {paired_model_idx}")
    paired_memberships = all_memberships[:, paired_model_idx]
    offline_a = 0.0
    max_auc = 0
    for test_a in np.arange(0, 1.1, 0.1):
        mia_scores = run_rmia(
            paired_model_idx,
            all_signals,
            population_signals,
            all_memberships,
            1,
            test_a,
        )
        fpr_list, tpr_list, _ = roc_curve(
            paired_memberships.ravel(), mia_scores.ravel()
        )
        roc_auc = auc(fpr_list, tpr_list)
        if roc_auc > max_auc:
            max_auc = roc_auc
            offline_a = test_a
            mia_scores_array = mia_scores.ravel().copy()
            membership_array = paired_memberships.ravel().copy()
        logger.info(f"offline_a={test_a:.2f}: AUC {roc_auc:.4f}")
    return offline_a, mia_scores_array, membership_array


def run_rmia(
    target_model_idx: int,
    all_signals: np.ndarray,
    population_signals: np.ndarray,
    all_memberships: np.ndarray,
    num_reference_models: int,
    offline_a: float = 0.0,
    online: bool = False,
) -> np.ndarray:
    """
    Attack a target model using RMIA in either offline or online mode.

    Offline: P(x) ≈ (1+a)/2 · P_out(x) + (1-a)/2  (uses only OUT reference models)
    Online:  P(x) = (P_in(x) + P_out(x)) / 2        (uses both IN and OUT reference models)

    Args:
        target_model_idx (int): Index of the target model.
        all_signals (np.ndarray): Softmax value of all samples in all models.
        population_signals (np.ndarray): Softmax value of population samples in all models.
        all_memberships (np.ndarray): Membership matrix for all models.
        num_reference_models (int): Number of reference models used for the attack.
        offline_a (float): Approximation coefficient for the offline setting (ignored when online=True).
        online (bool): If True, run the online variant using both IN and OUT reference models.

    Returns:
        np.ndarray: MIA score for all samples (higher = more likely to be a member).
    """
    target_signals = all_signals[:, target_model_idx]

    out_signals = get_rmia_out_signals(
        all_signals, all_memberships, target_model_idx, num_reference_models
    )
    mean_out_x = np.mean(out_signals, axis=1)

    if online:
        in_signals = get_rmia_in_signals(
            all_signals, all_memberships, target_model_idx, num_reference_models
        )
        mean_in_x = np.mean(in_signals, axis=1)
        mean_x = (mean_in_x + mean_out_x) / 2
    else:
        mean_x = (1 + offline_a) / 2 * mean_out_x + (1 - offline_a) / 2

    prob_ratio_x = target_signals.ravel() / mean_x

    # Population samples are always OUT for all reference models
    z_signals = population_signals[:, target_model_idx]
    population_memberships = np.zeros_like(population_signals).astype(bool)
    z_out_signals = get_rmia_out_signals(
        population_signals,
        population_memberships,
        target_model_idx,
        num_reference_models,
    )
    mean_out_z = np.mean(z_out_signals, axis=1)

    if online:
        # Population is never IN, so P(z) = P_out(z)
        mean_z = mean_out_z
    else:
        mean_z = (1 + offline_a) / 2 * mean_out_z + (1 - offline_a) / 2

    prob_ratio_z = z_signals.ravel() / mean_z

    ratios = prob_ratio_x[:, np.newaxis] / prob_ratio_z
    counts = np.average(ratios > 1.0, axis=1)

    return counts


def run_loss(target_signals: np.ndarray) -> np.ndarray:
    """
    Attack a target model using the LOSS attack.

    Args:
        target_signals (np.ndarray): Softmax value of all samples in the target model.

    Returns:
        np.ndarray: MIA score for all samples (a larger score indicates higher chance of being member).
    """
    mia_scores = -target_signals
    return mia_scores

def run_population_attack(
    target_signals: np.ndarray,
    population_signals: np.ndarray,
) -> np.ndarray:
    """
    Population attack: score = empirical CDF of the target signal within the population distribution.

    For each sample, computes the fraction of population samples whose confidence is
    lower than or equal to the sample's confidence. Higher confidence than most of the
    population -> higher score -> more likely to be a member.

    Args:
        target_signals (np.ndarray): Softmax probabilities from the target model. Shape: (n_samples,)
        population_signals (np.ndarray): Softmax probabilities from the target model on population data. Shape: (n_pop,)

    Returns:
        np.ndarray: Scores in [0, 1]. Higher = more likely member. Shape: (n_samples,)
    """
    pop_sorted = np.sort(population_signals.ravel())
    scores = np.searchsorted(pop_sorted, target_signals.ravel(), side="right") / len(pop_sorted)
    return scores


def run_lira(
    target_signals: np.ndarray,
    shadow_model_signals: np.ndarray,
    shadow_model_memberships: np.ndarray,
    target_memberships: np.ndarray,
    online: bool = False,
) -> np.ndarray:
    """
    Attack a target model using the LiRA (Likelihood-Ratio Attack) method.

    Offline: score = -log P(conf | N(μ_out, σ_out))
             Only OUT shadow models are used. Score is negated so that
             higher = more likely to be a member.

    Online:  score = log P(conf | N(μ_in, σ_in)) - log P(conf | N(μ_out, σ_out))
             Both IN and OUT shadow models are used. No negation needed.

    Args:
        target_signals (np.ndarray): Softmax signals from target model. Shape: (num_samples,)
        shadow_model_signals (np.ndarray): Softmax signals from shadow models.
                                           Shape: (num_samples, num_shadow_models)
        shadow_model_memberships (np.ndarray): Membership matrix for shadow models.
                                               Shape: (num_samples, num_shadow_models)
        target_memberships (np.ndarray): Membership array for target model. Shape: (num_samples,)
        online (bool): If True, run the online variant using both IN and OUT shadow models.

    Returns:
        np.ndarray: LiRA scores. Higher value = more likely to be a member. Shape: (num_samples,)
    """
    from scipy.stats import norm

    target_signals = target_signals.ravel().copy()
    target_memberships = target_memberships.ravel()

    # Replace NaN/Inf in signals
    finite_mask = np.isfinite(target_signals)
    if not finite_mask.all():
        fill = np.nanmedian(target_signals) if finite_mask.any() else 0.0
        target_signals[~finite_mask] = fill

    shadow_model_signals = shadow_model_signals.copy()
    shadow_finite = np.isfinite(shadow_model_signals)
    if not shadow_finite.all():
        col_medians = np.nanmedian(shadow_model_signals, axis=0)
        for j in range(shadow_model_signals.shape[1]):
            bad = ~shadow_finite[:, j]
            shadow_model_signals[bad, j] = col_medians[j] if np.isfinite(col_medians[j]) else 0.0

    n_samples = len(target_signals)

    score_mean_out = np.zeros(n_samples)
    score_std_out = np.zeros(n_samples)
    score_mean_in = np.zeros(n_samples)
    score_std_in = np.zeros(n_samples)

    for i in range(n_samples):
        sample_preds = shadow_model_signals[i, :]
        in_mask = shadow_model_memberships[i, :].astype(bool)
        out_mask = ~in_mask

        out_preds = sample_preds[out_mask]
        if len(out_preds) > 0:
            score_mean_out[i] = np.mean(out_preds)
            score_std_out[i] = np.std(out_preds)
        else:
            score_mean_out[i] = np.mean(sample_preds)
            score_std_out[i] = 1e-8

        if online:
            in_preds = sample_preds[in_mask]
            if len(in_preds) > 0:
                score_mean_in[i] = np.mean(in_preds)
                score_std_in[i] = np.std(in_preds)
            else:
                score_mean_in[i] = np.mean(sample_preds)
                score_std_in[i] = 1e-8

    # Global std: mean of per-sample stds (more stable than per-sample variance)
    global_std_out = np.maximum(np.mean(score_std_out), 1e-8)

    if online:
        global_std_in = np.maximum(np.mean(score_std_in), 1e-8)
        lira_scores = (
            norm.logpdf(target_signals, score_mean_in, global_std_in)
            - norm.logpdf(target_signals, score_mean_out, global_std_out)
        )
    else:
        lira_scores = norm.logpdf(target_signals, score_mean_out, global_std_out)
        # Negate: lower log-likelihood under OUT → more likely to be a member
        lira_scores = -lira_scores

    return lira_scores


def get_shadow_model_signals(
    all_signals: np.ndarray,
    all_memberships: np.ndarray,
    target_model_idx: int,
    num_reference_models: int,
) -> tuple:
    """
    Extract signals from shadow models (all models except target and paired model).
    Also returns the corresponding membership matrix for shadow models.
    
    Args:
        all_signals (np.ndarray): Signals from all models.
                                  Shape: (num_samples, num_models)
        all_memberships (np.ndarray): Membership matrix for all models.
                                      Shape: (num_samples, num_models)
        target_model_idx (int): Index of the target model.
        num_reference_models (int): Number of shadow models to use.

    Returns:
        tuple: (shadow_signals, shadow_memberships)
               - shadow_signals: Shape (num_samples, num_shadow_models)
               - shadow_memberships: Shape (num_samples, num_shadow_models)
    """
    paired_model_idx = (
        target_model_idx + 1 if target_model_idx % 2 == 0 else target_model_idx - 1
    )
    
    # Get indices of shadow models (excluding target and its paired model)
    shadow_indices = [
        i
        for i in range(all_signals.shape[1])
        if i != target_model_idx and i != paired_model_idx
    ][: 2 * num_reference_models]
    
    # Extract shadow model signals and memberships
    shadow_signals = all_signals[:, shadow_indices]
    shadow_memberships = all_memberships[:, shadow_indices]
    
    return shadow_signals, shadow_memberships