from typing import Any

import numpy as np
from sklearn.metrics import auc, roc_curve


def get_rmia_out_signals(
    all_signals: np.ndarray,
    all_memberships: np.ndarray,
    target_model_idx: int,
    num_reference_models: int,
) -> np.ndarray:
    """
    Get average prediction probability of samples over offline reference models (excluding the target model).

    Args:
        all_signals (np.ndarray): Softmax value of all samples in every model.
        all_memberships (np.ndarray): Membership matrix for all models (if a sample is used for training a model).
        target_model_idx (int): Target model index.
        num_reference_models (int): Number of reference models used for the attack.

    Returns:
        np.ndarray: Average softmax value for each sample over OUT reference models.
    """
    paired_model_idx = (
        target_model_idx + 1 if target_model_idx % 2 == 0 else target_model_idx - 1
    )
    # Add non-target and non-paired model indices
    columns = [
        i
        for i in range(all_signals.shape[1])
        if i != target_model_idx and i != paired_model_idx
    ][: 2 * num_reference_models]
    selected_signals = all_signals[:, columns]
    non_members = ~all_memberships[:, columns]
    out_signals = selected_signals * non_members
    # Sort the signals such that only the non-zero signals (out signals) are kept
    out_signals = -np.sort(-out_signals, axis=1)[:, :num_reference_models]
    return out_signals


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
    offline_a: float,
) -> np.ndarray:
    """
    Attack a target model using the RMIA attack with the help of offline reference models.

    Args:
        target_model_idx (int): Index of the target model.
        all_signals (np.ndarray): Softmax value of all samples in the target model.
        population_signals (np.ndarray): Softmax value of all population samples in the target model.
        all_memberships (np.ndarray): Membership matrix for all models.
        num_reference_models (int): Number of reference models used for the attack.
        offline_a (float): Coefficient offline_a is used to approximate p(x) using P_out in the offline setting.

    Returns:
        np.ndarray: MIA score for all samples (a larger score indicates higher chance of being member).
    """
    target_signals = all_signals[:, target_model_idx]
    out_signals = get_rmia_out_signals(
        all_signals, all_memberships, target_model_idx, num_reference_models
    )
    mean_out_x = np.mean(out_signals, axis=1)
    mean_x = (1 + offline_a) / 2 * mean_out_x + (1 - offline_a) / 2
    prob_ratio_x = target_signals.ravel() / mean_x

    z_signals = population_signals[:, target_model_idx]
    population_memberships = np.zeros_like(population_signals).astype(
        bool
    )  # All population data are OUT for all models
    z_out_signals = get_rmia_out_signals(
        population_signals,
        population_memberships,
        target_model_idx,
        num_reference_models,
    )
    mean_out_z = np.mean(z_out_signals, axis=1)
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

def run_lira(
    target_signals: np.ndarray,
    shadow_model_signals: np.ndarray,
    shadow_model_memberships: np.ndarray,
    target_memberships: np.ndarray,
) -> np.ndarray:
    """
    Attack a target model using the LiRA (Likelihood-Ratio Attack) method.
    
    Aligned with PrivacyGuard's offline LiRA implementation.
    
    For each sample x, we:
    1. Compute predictions from shadow models where sample was NOT in training (OUT)
    2. Estimate per-sample μ_out (mean of OUT predictions)
    3. Estimate global σ_out (mean of per-sample standard deviations)
    4. Compute: score = log P(conf_obs | N(μ_out, σ²_out_global))
    5. Negate for offline variant on hold-out test set
    
    The intuition: Members typically have predictions that fit better under the 
    "non-member" (OUT) distribution than population data.

    Args:
        target_signals (np.ndarray): Softmax signals from target model.
                                     Shape: (num_samples,)
        shadow_model_signals (np.ndarray): Softmax signals from shadow models.
                                           Shape: (num_samples, num_shadow_models)
        shadow_model_memberships (np.ndarray): Membership matrix for shadow models.
                                               Shape: (num_samples, num_shadow_models)
                                               Each entry [i,j] = 1 if sample i was in shadow model j's training
        target_memberships (np.ndarray): Membership array for target model (1=in, 0=out).
                                         Shape: (num_samples,)
       
    Returns:
        np.ndarray: LiRA scores for all samples.
                   For offline LiRA: -log P(conf_obs | N(μ_out, σ²_out_global))
                   Higher value = more likely to be a member
                   Shape: (num_samples,)
    """
    from scipy.stats import norm
    
    target_signals = target_signals.ravel()
    target_memberships = target_memberships.ravel()
    n_samples = len(target_signals)
    
    # For each sample, compute μ of OUT predictions (per-sample)
    # OUT predictions = predictions from shadow models where sample was NOT in training
    score_mean_out = np.zeros(n_samples)
    score_std_out = np.zeros(n_samples)
    
    # shadow_model_memberships[i, j] = 1 if sample i was in shadow model j's training
    
    for i in range(n_samples):
        sample_shadow_preds = shadow_model_signals[i, :]  # Predictions from all shadows
        sample_memberships = shadow_model_memberships[i, :]  # Which shadows included this sample
        
        # OUT predictions: from shadows where sample was NOT included
        out_mask = ~sample_memberships.astype(bool)
        out_preds = sample_shadow_preds[out_mask]
        
        if len(out_preds) > 0:
            score_mean_out[i] = np.mean(out_preds)
            score_std_out[i] = np.std(out_preds)
        else:
            # No OUT predictions available, use overall mean and small std
            score_mean_out[i] = np.mean(sample_shadow_preds)
            score_std_out[i] = 1e-8
    
    # Compute global standard deviation (mean of all per-sample stds)
    global_std_out = np.mean(score_std_out)
    global_std_out = np.maximum(global_std_out, 1e-8)
    
    # For offline LiRA: compute log-likelihood under OUT distribution
    # log P(conf_obs | N(μ_out_sample, σ²_out_global))
    lira_scores = norm.logpdf(target_signals, score_mean_out, global_std_out)
    
    # Negate scores for offline LiRA on hold-out test set
    # (Aligns with PrivacyGuard: if not online_attack, negate the scores)
    # Higher negated score = more likely to be a member
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