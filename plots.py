from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray 


def plot_predictions_vs_actual(
    hist_time: list[Any],
    hist_actual: NDArray[np.float64],
    future_time: list[Any],
    future_pred: NDArray[np.float64],
    train_actual: NDArray[np.float64],
    train_fitted: NDArray[np.float64],
    path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(hist_time, hist_actual, lw=0.8, label="actual (history)")
    axes[0].plot(
        future_time, future_pred, lw=1.3, color="tab:red", label="forecast"
    )
    axes[0].set_title("Power: history vs future forecast")
    axes[0].set_xlabel("time (UTC)")
    axes[0].set_ylabel("total power [MW]")
    axes[0].tick_params(axis="x", labelrotation=45)
    axes[0].legend()

    axes[1].scatter(train_actual, train_fitted, s=6, alpha=0.3)
    lo = float(min(train_actual.min(), train_fitted.min()))
    hi = float(max(train_actual.max(), train_fitted.max()))
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[1].set_title("Fitted vs actual (train)")
    axes[1].set_xlabel("actual power [MW]")
    axes[1].set_ylabel("predicted power [MW]")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cv_folds(folds: list[dict[str, Any]], path: str) -> None:
    n = max(len(folds), 1)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)
    for i, (ax, fold) in enumerate(zip(axes[:, 0], folds), start=1):
        ax.plot(fold["time"], fold["y_true"], lw=0.8, label="actual")
        ax.plot(
            fold["time"],
            fold["y_pred"],
            lw=0.8,
            color="tab:red",
            label="predicted",
        )
        ax.set_title(f"Fold {i} (n={len(fold['y_true'])})")
        ax.set_xlabel("time (UTC)")
        ax.set_ylabel("total power [MW]")
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
