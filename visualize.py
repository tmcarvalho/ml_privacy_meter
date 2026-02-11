import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
import pandas as pd
from typing import Tuple, Optional, List


def plot_roc(fpr_list, tpr_list, roc_auc, path):
    """Function to get the ROC plot using FPR and TPR results

    Args:
        fpr_list (list or ndarray): List of FPR values
        tpr_list (list or ndarray): List of TPR values
        roc_auc (float or floating): Area Under the ROC Curve
        path (str): Folder for saving the ROC plot
    """
    range01 = np.linspace(0, 1)
    plt.fill_between(fpr_list, tpr_list, alpha=0.15)
    plt.plot(fpr_list, tpr_list, label="ROC curve")
    plt.plot(range01, range01, "--", label="Random guess")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.legend()
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.title("ROC curve")
    plt.text(
        0.7,
        0.3,
        f"AUC = {roc_auc:.03f}",
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.savefig(
        fname=path,
        dpi=1000,
    )
    plt.clf()


def plot_roc_log(fpr_list, tpr_list, roc_auc, path):
    """Function to get the log-scale ROC plot using FPR and TPR results

    Args:
        fpr_list (list or ndarray): List of False Positive Rate values
        tpr_list (list or ndarray): List of True Positive Rate values
        roc_auc (float or floating): Area Under the ROC Curve
        path (str): Folder for saving the ROC plot
    """
    range01 = np.linspace(0, 1)
    plt.fill_between(fpr_list, tpr_list, alpha=0.15)
    plt.plot(fpr_list, tpr_list, label="ROC curve")
    plt.plot(range01, range01, "--", label="Random guess")
    plt.xlim([10e-6, 1])
    plt.ylim([10e-6, 1])
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.title("ROC curve")
    plt.text(
        0.7,
        0.3,
        f"AUC = {roc_auc:.03f}",
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.savefig(
        fname=path,
        dpi=1000,
    )
    plt.clf()


def plot_eps_vs_num_guesses(
    eps_list, correct_num_list, k_neg_list, k_pos_list, total_num, path
):
    """Function to get the auditing performance versus number of guesses plot

    Args:
        eps_list (list or ndarray): List of audited eps values
        correct_num_list (list or ndarray): List of number of correct guesses
        k_neg_list (list or ndarray): List of positive guesses
        k_pos_list (list or ndarray): List of negative guesses
        total_num (int): Total number of samples
        path (str): Folder for saving the auditing performance plot
    """
    fig, ax = plt.subplots(1, 1)
    num_guesses_grid = np.array(k_neg_list) + total_num - np.array(k_pos_list)
    ax.scatter(
        num_guesses_grid,
        correct_num_list / num_guesses_grid,
        color="#FF9999",
        alpha=0.6,
        label=r"Inference Accuracy",
        s=80,
    )
    ax.scatter(
        num_guesses_grid, eps_list, color="#66B2FF", alpha=0.6, label=r"$EPS LB$", s=80
    )
    ax.set_xlabel(r"number of guesses")
    plt.legend(fontsize=10)

    min_interval_idx = np.argmax(eps_list)
    t = f"k_neg={k_neg_list[min_interval_idx]} and k_pos={k_pos_list[min_interval_idx]} enables the highest audited EPS LB: num of guesses is {num_guesses_grid[min_interval_idx]}, EPS LB is {eps_list[min_interval_idx]}"
    tt = textwrap.fill(t, width=70)
    plt.text(num_guesses_grid.mean(), -0.2, tt, ha="center", va="top")

    plt.savefig(path, bbox_inches="tight")

    plt.close()


def plot_score_distributions(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    title: str = "Score Distributions",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot boxplot and histogram of scores for members vs non-members.

    Args:
        train_scores (ndarray): Scores for training data (members)
        test_scores (ndarray): Scores for test data (non-members)
        title (str): Title for the plots
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot
    boxplot_data = [train_scores, test_scores]
    bp = axes[0].boxplot(
        boxplot_data,
        labels=["Training Data\n(Members)", "Test Data\n(Non-members)"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], ["tab:orange", "tab:blue"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].set_title(f"{title} - Boxplot", fontsize=12, fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Histogram
    axes[1].hist(
        train_scores,
        bins=30,
        alpha=0.6,
        label="Training Data (Members)",
        color="tab:orange",
        edgecolor="black",
    )
    axes[1].hist(
        test_scores,
        bins=30,
        alpha=0.6,
        label="Test Data (Non-members)",
        color="tab:blue",
        edgecolor="black",
    )
    axes[1].set_xlabel("Score", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title(f"{title} - Histogram", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_membership_statistics(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    score_column: str = "score",
    title: str = "Membership Statistics",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot bar charts showing count and statistics for members vs non-members.

    Args:
        df_train (DataFrame): DataFrame with training data (members)
        df_test (DataFrame): DataFrame with test data (non-members)
        score_column (str): Name of the score column
        title (str): Title for the plot
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Count of members vs non-members
    counts = [len(df_train), len(df_test)]
    labels = ["Members\n(Training)", "Non-members\n(Test)"]
    colors = ["tab:orange", "tab:blue"]
    bars = axes[0, 0].bar(labels, counts, color=colors, alpha=0.7, edgecolor="black")
    axes[0, 0].set_ylabel("Count", fontsize=11)
    axes[0, 0].set_title("Sample Counts", fontsize=12, fontweight="bold")
    axes[0, 0].grid(True, axis="y", alpha=0.3)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # 2. Mean scores
    mean_scores = [df_train[score_column].mean(), df_test[score_column].mean()]
    bars = axes[0, 1].bar(labels, mean_scores, color=colors, alpha=0.7, edgecolor="black")
    axes[0, 1].set_ylabel("Mean Score", fontsize=11)
    axes[0, 1].set_title("Mean Score Comparison", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, axis="y", alpha=0.3)
    for bar, score in zip(bars, mean_scores):
        height = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # 3. Standard deviation
    std_scores = [df_train[score_column].std(), df_test[score_column].std()]
    bars = axes[1, 0].bar(labels, std_scores, color=colors, alpha=0.7, edgecolor="black")
    axes[1, 0].set_ylabel("Standard Deviation", fontsize=11)
    axes[1, 0].set_title("Score Variability", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    for bar, std in zip(bars, std_scores):
        height = bar.get_height()
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # 4. Min and Max scores
    min_scores = [df_train[score_column].min(), df_test[score_column].min()]
    max_scores = [df_train[score_column].max(), df_test[score_column].max()]
    x = np.arange(len(labels))
    width = 0.35
    bars1 = axes[1, 1].bar(x - width / 2, min_scores, width, label="Min Score", color="lightcoral", alpha=0.7, edgecolor="black")
    bars2 = axes[1, 1].bar(x + width / 2, max_scores, width, label="Max Score", color="lightgreen", alpha=0.7, edgecolor="black")
    axes[1, 1].set_ylabel("Score Range", fontsize=11)
    axes[1, 1].set_title("Score Range (Min/Max)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_attack_results_summary(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    score_column: str = "score",
    attack_name: str = "RMIA",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[float, float, float]:
    """Plot comprehensive attack results with members vs non-members analysis.

    Args:
        df_train (DataFrame): DataFrame with training data (members)
        df_test (DataFrame): DataFrame with test data (non-members)
        score_column (str): Name of the score column
        attack_name (str): Name of the attack
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the plot

    Returns:
        Tuple[float, float, float]: (mean_train_score, mean_test_score, accuracy)
    """
    train_scores = df_train[score_column].values
    test_scores = df_test[score_column].values

    # Calculate accuracy (percentage of correct predictions)
    # Members: score > 0.5, Non-members: score <= 0.5
    correct_members = np.sum(train_scores > 0.5)
    correct_non_members = np.sum(test_scores <= 0.5)
    accuracy = (correct_members + correct_non_members) / (len(train_scores) + len(test_scores))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Score distributions with overlays
    axes[0, 0].hist(
        train_scores,
        bins=40,
        alpha=0.6,
        label="Members (Training)",
        color="tab:orange",
        density=True,
        edgecolor="black",
    )
    axes[0, 0].hist(
        test_scores,
        bins=40,
        alpha=0.6,
        label="Non-members (Test)",
        color="tab:blue",
        density=True,
        edgecolor="black",
    )
    axes[0, 0].axvline(0.5, color="red", linestyle="--", linewidth=2, label="Decision Threshold (0.5)")
    axes[0, 0].set_xlabel("Score", fontsize=11)
    axes[0, 0].set_ylabel("Density", fontsize=11)
    axes[0, 0].set_title("Score Distributions (Normalized)", fontsize=12, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Classification breakdown
    categories = ["Members\nCorrectly Identified", "Members\nMisidentified", 
                  "Non-members\nCorrectly Identified", "Non-members\nMisidentified"]
    counts = [correct_members, len(train_scores) - correct_members,
              correct_non_members, len(test_scores) - correct_non_members]
    colors_breakdown = ["darkgreen", "darkred", "darkblue", "orange"]
    bars = axes[0, 1].bar(range(len(categories)), counts, color=colors_breakdown, alpha=0.7, edgecolor="black")
    axes[0, 1].set_ylabel("Count", fontsize=11)
    axes[0, 1].set_title("Classification Breakdown", fontsize=12, fontweight="bold")
    axes[0, 1].set_xticks(range(len(categories)))
    axes[0, 1].set_xticklabels(categories, fontsize=9)
    axes[0, 1].grid(True, axis="y", alpha=0.3)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 3. Performance metrics
    tpr = correct_members / len(train_scores)  # True Positive Rate
    fpr = (len(test_scores) - correct_non_members) / len(test_scores)  # False Positive Rate
    metrics_labels = ["TPR\n(Members Found)", "FPR\n(False Positives)"]
    metrics_values = [tpr, fpr]
    bars = axes[1, 0].bar(metrics_labels, metrics_values, color=["green", "red"], alpha=0.7, edgecolor="black")
    axes[1, 0].set_ylabel("Rate", fontsize=11)
    axes[1, 0].set_title("Performance Metrics", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # 4. Summary statistics
    axes[1, 1].axis("off")
    summary_text = f"""
{attack_name} Attack Results Summary

Members (Training Data):
  • Count: {len(train_scores)}
  • Mean Score: {train_scores.mean():.4f}
  • Std Dev: {train_scores.std():.4f}
  • Min Score: {train_scores.min():.4f}
  • Max Score: {train_scores.max():.4f}

Non-members (Test Data):
  • Count: {len(test_scores)}
  • Mean Score: {test_scores.mean():.4f}
  • Std Dev: {test_scores.std():.4f}
  • Min Score: {test_scores.min():.4f}
  • Max Score: {test_scores.max():.4f}

Attack Performance:
  • True Positive Rate: {tpr:.4f}
  • False Positive Rate: {fpr:.4f}
  • Overall Accuracy: {accuracy:.4f}
    """
    axes[1, 1].text(
        0.1,
        0.95,
        summary_text,
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle(f"{attack_name} - Members vs Non-members Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    return train_scores.mean(), test_scores.mean(), accuracy
