"""
Markov-chain stock model (5 states) per proposal:

S1: Significant Increase (daily % change >= +3%)
S2: Slight Increase      (+0.5% to +3%)
S3: Stable               (-0.5% to +0.5%)
S4: Slight Decrease      (-3% to -0.5%)
S5: Significant Decrease (<= -3%)

Data: Yahoo Finance daily adjusted close (recommended) or close.
Outputs:
- transition matrix P
- stationary distribution pi (long-run state probabilities)
- simple "volatility via transitions" indicators
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# 5-state definitions (proposal)
# ----------------------------
STATE_NAMES = ["S1_sig_up", "S2_slight_up", "S3_stable", "S4_slight_down", "S5_sig_down"]
STATE_TO_IDX = {s: i for i, s in enumerate(STATE_NAMES)}


# ----------------------------
# Load price data from CSV
# ----------------------------
def load_prices_from_csv(filepath: str, prefer_adjusted: bool = True) -> pd.Series:
    """
    Loads a Yahoo Finance-style CSV and returns a 1D price Series.
    Prefers 'Adj Close' if available, otherwise uses 'Close'.

    Expected columns (typical):
    Date, Open, High, Low, Close, Adj Close, Volume
    """
    df = pd.read_csv(filepath)

    # Choose price column
    if prefer_adjusted and "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    elif "Adj Close" in df.columns:
        price_col = "Adj Close"
    else:
        raise ValueError("CSV must contain a 'Close' or 'Adj Close' column.")

    prices = df[price_col].dropna().astype(float)

    # If Date exists, use it as index (nice but not required)
    if "Date" in df.columns:
        try:
            dates = pd.to_datetime(df["Date"])
            prices.index = dates[prices.index]
            prices = prices.sort_index()
        except Exception:
            # If date parsing fails, still proceed with numeric index
            pass

    prices.name = price_col
    return prices


# ----------------------------
# Returns -> state mapping
# ----------------------------
def pct_change_percent(prices: pd.Series) -> pd.Series:
    """Daily percent change in percent units (e.g., 1.2 means +1.2%)."""
    return prices.pct_change() * 100.0


def assign_state(r) -> str:
    """
    Map a daily return r (%) to states:
      S1: >= +3%
      S2: +0.5% to < +3%
      S3: -0.5% to < +0.5%
      S4: > -3% to <= -0.5%
      S5: <= -3%
    """
    if pd.isna(r):
        return np.nan  # type: ignore

    r = float(r)

    if r >= 3.0:
        return "S1_sig_up"
    if 0.5 <= r < 3.0:
        return "S2_slight_up"
    if -0.5 < r < 0.5:
        return "S3_stable"
    if -3.0 < r <= -0.5:
        return "S4_slight_down"
    return "S5_sig_down"


def build_state_series(prices: pd.Series) -> pd.Series:
    rets = pct_change_percent(prices)
    states = rets.map(assign_state).dropna()
    states.name = "state"
    return states


# ----------------------------
# Transition matrix
# ----------------------------
def transition_counts(states: pd.Series) -> np.ndarray:
    """C[i,j] = number of transitions from state i to state j."""
    s = states.to_numpy()
    C = np.zeros((5, 5), dtype=int)

    for a, b in zip(s[:-1], s[1:]):
        i = STATE_TO_IDX[a]
        j = STATE_TO_IDX[b]
        C[i, j] += 1

    return C


def transition_matrix(states: pd.Series) -> np.ndarray:
    """Row-stochastic transition matrix P from transition counts."""
    C = transition_counts(states)
    row_sums = C.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(row_sums > 0, C / row_sums, 0.0)

    return P


# ----------------------------
# Stationary distribution
# ----------------------------
def stationary_distribution(P: np.ndarray, tol: float = 1e-12, max_iter: int = 200000) -> np.ndarray:
    """
    Power iteration to find pi such that pi = pi P.
    Returns best-effort even if not perfectly converged.
    """
    n = P.shape[0]
    pi = np.ones(n) / n

    for _ in range(max_iter):
        pi_next = pi @ P
        if np.linalg.norm(pi_next - pi, ord=1) < tol:
            return pi_next
        pi = pi_next

    return pi



# ----------------------------
# Heatmap of the estimated transition matrix
# ----------------------------
def plot_transition_heatmap(P: np.ndarray):
    fig, ax = plt.subplots()
    im = ax.imshow(P)

    ax.set_xticks(range(len(STATE_NAMES)))
    ax.set_yticks(range(len(STATE_NAMES)))
    ax.set_xticklabels(STATE_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(STATE_NAMES)

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax.text(j, i, f"{P[i, j]:.2f}",
                    ha="center", va="center", color="black")

    ax.set_title("Transition Matrix Heatmap")
    ax.set_xlabel("Next Day State")
    ax.set_ylabel("Current Day State")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig("transition_heatmap.png")
    plt.close()

# ----------------------------
# Stationary distribution map
# ----------------------------
def plot_stationary_distribution(pi: np.ndarray):
    plt.figure()
    plt.bar(STATE_NAMES, pi)
    plt.title("Stationary Distribution of Return States")
    plt.ylabel("Long-Run Probability")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("stationary_distribution.png")
    plt.close()





def as_dataframe(P: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(P, index=STATE_NAMES, columns=STATE_NAMES)


# ----------------------------
# Main runner
# ----------------------------
def run_model(csv_path: str = "AAPL.csv") -> None:
    prices = load_prices_from_csv(csv_path, prefer_adjusted=True)
    states = build_state_series(prices)

    P = transition_matrix(states)
    pi = stationary_distribution(P)

    print(f"\nLoaded: {csv_path}")
    print(f"Price column: {prices.name}")
    print(f"Days classified into states: {len(states)}")

    print("\nTransition Matrix P (rows sum to 1):")
    print(as_dataframe(P).round(4))

    print("\nStationary distribution π (long-run state probabilities):")
    for name, val in zip(STATE_NAMES, pi):
        print(f"  {name:14s}: {val:.4f}")


    plot_transition_heatmap(P)
    plot_stationary_distribution(pi)




if __name__ == "__main__":
    run_model("AAPL.csv")
