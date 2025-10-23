#!/usr/bin/env python3
"""
plot_command_latency.py

Compute and plot the latency between a commanded value and the
actual/joystick state reaching that same value.

Figure:
    • Histogram of latency values ONLY (no time axis)
    • Overlaid Normal-fit curve (μ, σ) — shown in **milliseconds**

Wiring (hardcoded):
    TIME   = "__time"
    ACTUAL = "/joystick_state/data"
    CMD    = "/move_arm/data"

Action values (with tolerance):
    UP     = 0.037999999
    NO-OP  = 0.029999999
    DOWN   = 0.022

Outputs:
    • PNG:  OUT_PNG
    • CSV:  OUT_CSV  (filtered latencies in seconds; plotting converts to ms)
    • Console: stats in ms + first latency (ms) + first few events for debugging
"""

import math
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

# ============================ USER SETTINGS =====================================

# Hardcode your CSV path here:
CSV_PATH = "latencies-eth.csv"   # <-- change this to your file, e.g. "/mnt/data/bag.csv"

# Hardcode your column headers exactly as they appear in the CSV:
TIME_COL = "__time"                 # time column
ACTUAL_COL = "/joystick_state/data" # actual signal (joystick)
COMMANDED_COL = "/move_arm/data"    # commanded signal

# Output artifacts:
OUT_PNG = "latency_histogram_ethernet.png"
OUT_CSV = "latencies.csv"

# Histogram settings
NUM_BINS = 100           # more bins = finer resolution
CLIP_NEGATIVES = False  # defensive: drop negative latencies (shouldn't occur)

# Latency filtering
MIN_LATENCY_MS = 100.0  # << Ignore (drop) commands with latency < 100 ms

# Debug printing: how many early command-change events to print (after filtering)
DEBUG_FIRST_N_EVENTS = 5

# =================================================================================

# ---- Action mapping and utilities ------------------------------------------------

ACTION_FLOATS = {
    "DOWN": 0.022,
    "NO-OP": 0.029999999,
    "UP": 0.037999999,
}

# Tolerance for matching floats from file to canonical values
EPS = 5e-4  # tolerance for float equality vs. canonical action values


def float_to_action(x: float, eps: float = EPS) -> Optional[str]:
    """Map a float value to an action label with tolerance."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    for label, target in ACTION_FLOATS.items():
        if abs(float(x) - target) <= eps:
            return label
    return None


def series_to_actions(s: pd.Series) -> pd.Series:
    """Convert a float Series to action labels using float_to_action."""
    return s.map(float_to_action)


# ---- Latency computation ---------------------------------------------------------

def compute_command_latencies(
    time_s: np.ndarray,
    actual_actions: List[Optional[str]],
    commanded_actions: List[Optional[str]],
) -> pd.DataFrame:
    """
    For each change in commanded_actions, compute how long it takes until the
    actual_actions matches the newly commanded action.

    Rules:
      • Treat a "new command" only when the command value is NOT None and differs
        from the last non-None command.
      • "Instant match" ONLY if the actual AT THE SAME ROW already equals the new command.
      • Otherwise, scan forward to the first row where actual == command.

    Returns a DataFrame with columns:
        - t_command (sec), commanded, t_match (sec), latency_s, match_index, command_index
    """
    assert len(time_s) == len(actual_actions) == len(commanded_actions)
    n = len(time_s)

    rows = []
    prev_cmd_non_none: Optional[str] = None

    for i in range(n):
        cmd = commanded_actions[i]
        act = actual_actions[i]

        # Only consider a "new command" when cmd is NOT None and different from the last NON-None command
        is_new_cmd = (cmd is not None) and (cmd != prev_cmd_non_none)

        if is_new_cmd:
            t0 = time_s[i]

            # --- Instant-match guard (same row only) ---
            if (act is not None) and (act == cmd):
                rows.append(
                    {
                        "command_index": i,
                        "t_command": float(t0),
                        "commanded": cmd,
                        "match_index": i,
                        "t_match": float(t0),
                        "latency_s": 0.0,
                    }
                )
            else:
                # Find earliest j >= i where actual matches cmd
                match_idx = -1
                for j in range(i, n):
                    if actual_actions[j] == cmd:
                        match_idx = j
                        break

                if match_idx >= 0:
                    t_match = time_s[match_idx]
                    latency = t_match - t0
                else:
                    t_match = np.nan
                    latency = np.nan

                rows.append(
                    {
                        "command_index": i,
                        "t_command": float(t0),
                        "commanded": cmd,
                        "match_index": match_idx,
                        "t_match": float(t_match) if not np.isnan(t_match) else np.nan,
                        "latency_s": float(latency) if not np.isnan(latency) else np.nan,
                    }
                )

            # Update the last NON-None command only when we see a real command value
            prev_cmd_non_none = cmd

        # IMPORTANT: do NOT overwrite prev_cmd_non_none when cmd is None.
        # That prevents blanks from creating fake "new command" events later.

    return pd.DataFrame(rows)


# ---- Plotting: histogram with normal-fit curve (in ms) ---------------------------

def _apply_dense_ticks(ax, data_min: float, data_max: float, target_major_count: int = 16):
    """
    Add dense major and minor ticks on the x-axis.
    """
    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_max <= data_min:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=target_major_count))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        return

    span = data_max - data_min
    if span <= 0:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=target_major_count))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        return

    # Compute a "nice" step for majors (aim ~target_major_count)
    raw_step = span / target_major_count
    # Round raw_step to a 1-2-5 * 10^k grid
    exp = 10 ** np.floor(np.log10(raw_step))
    mant = raw_step / exp
    if mant < 1.5:
        step = 1 * exp
    elif mant < 3.5:
        step = 2 * exp
    elif mant < 7.5:
        step = 5 * exp
    else:
        step = 10 * exp

    # Create ticks aligned to the data_min
    first_tick = step * np.ceil(data_min / step)
    majors = np.arange(first_tick, data_max + 0.5 * step, step)

    ax.set_xticks(majors)
    # Minor ticks: 5 per major step
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))


def plot_latency_histogram_ms(lat_df: pd.DataFrame, out_png: Optional[str] = None):
    """
    Plot latency histogram (milliseconds) with Normal-fit curve (μ, σ in ms).
    Also prints summary stats and the FIRST latency value (ms) for debugging.
    """
    # Convert to milliseconds
    lat_ms = (lat_df["latency_s"].to_numpy(dtype=float)) * 1000.0
    lat_ms = lat_ms[np.isfinite(lat_ms)]

    if CLIP_NEGATIVES:
        lat_ms = lat_ms[lat_ms >= 0]

    if lat_ms.size == 0:
        print("No valid latencies to plot.")
        return

    # Print the first latency value (ms) for debugging
    first_latency_ms = lat_ms[0]
    print(f"[Debug] First latency value (ms): {first_latency_ms:.6f}")

    # Stats in ms
    mu = float(np.mean(lat_ms))
    sigma = float(np.std(lat_ms, ddof=1)) if lat_ms.size > 1 else 0.0
    median = float(np.median(lat_ms))
    lmin, lmax = float(np.min(lat_ms)), float(np.max(lat_ms))

    print(f"Latency (ms) count={lat_ms.size}, mean={mu:.6f} ms, median={median:.6f} ms, "
          f"std={sigma:.6f} ms, min={lmin:.6f} ms, max={lmax:.6f} ms")

    # Histogram (density=True so the PDF curve is on the same scale)
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    counts, bins, patches = ax.hist(lat_ms, bins=NUM_BINS, density=True, alpha=0.5, edgecolor="black")

    # Normal-fit curve in ms
    if sigma > 0:
        x = np.linspace(bins[0], bins[-1], 1024)
        norm_pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, norm_pdf, linewidth=2, label=f"Normal fit μ={mu:.3f} ms, σ={sigma:.3f} ms")

    # Labels
    ax.set_title("Latency Distribution (command → actual)")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Dense ticks
    _apply_dense_ticks(ax, data_min=bins[0], data_max=bins[-1], target_major_count=16)

    if out_png:
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        print(f"[Saved] {out_png}")
    else:
        plt.show()


# ---- Filtering helper ------------------------------------------------------------

def filter_min_latency_ms(lat_df: pd.DataFrame, min_latency_ms: float) -> pd.DataFrame:
    """
    Keep only events with latency >= min_latency_ms (in milliseconds).
    """
    if lat_df.empty:
        return lat_df
    lat_ms = lat_df["latency_s"] * 1000.0
    keep = lat_ms >= float(min_latency_ms)
    return lat_df[keep].reset_index(drop=True)


# ---- Main (no argparse; uses hardcoded paths & column names) ---------------------

def main():
    # Read CSV with flexible separator and headers
    try:
        df = pd.read_csv(CSV_PATH, sep=None, engine="python", comment="#")
    except Exception:
        df = pd.read_csv(CSV_PATH, sep=r"\s+", engine="python", comment="#")

    if df.empty or len(df.columns) < 2:
        raise ValueError("CSV appears empty or lacks required columns.")

    # Enforce explicit column names (prevents swapped signals)
    for col in (TIME_COL, ACTUAL_COL, COMMANDED_COL):
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found. "
                f"Columns present: {list(df.columns)}"
            )

    # Coerce and clean
    time_s = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(time_s)):
        raise ValueError("Non-finite values found in the time column.")

    actual_vals = pd.to_numeric(df[ACTUAL_COL], errors="coerce")
    commanded_vals = pd.to_numeric(df[COMMANDED_COL], errors="coerce")

    # Map floats -> action labels
    actual_actions = series_to_actions(actual_vals)
    commanded_actions = series_to_actions(commanded_vals)

    # Warn if unmapped values exist
    n_unmapped_actual = int(actual_actions.isna().sum())
    n_unmapped_cmd = int(commanded_actions.isna().sum())
    if n_unmapped_actual or n_unmapped_cmd:
        print(f"[Warning] Unmapped values — actual: {n_unmapped_actual}, commanded: {n_unmapped_cmd}. "
              f"Adjust EPS={EPS} if needed.")

    # Compute latencies per commanded change (in seconds)
    lat_df_all = compute_command_latencies(
        time_s=time_s,
        actual_actions=actual_actions.tolist(),
        commanded_actions=commanded_actions.tolist(),
    )

    # Filter by minimum latency (ms)
    lat_df = filter_min_latency_ms(lat_df_all, MIN_LATENCY_MS)

    # Report drops
    dropped = 0 if lat_df_all is None else (len(lat_df_all) - len(lat_df))
    print(f"[Info] Applied min-latency filter: ≥ {MIN_LATENCY_MS:.0f} ms "
          f"(kept {len(lat_df)} of {len(lat_df_all)}; dropped {dropped}).")

    # Extra debug: show first few filtered command changes with their match and latency (ms)
    if not lat_df.empty:
        print("\n[Debug] First few command changes (filtered):")
        for _, row in lat_df.head(DEBUG_FIRST_N_EVENTS).iterrows():
            t_cmd = row["t_command"]
            t_match = row["t_match"]
            lat_ms = row["latency_s"] * 1000.0 if np.isfinite(row["latency_s"]) else float('nan')
            print(f"  cmd='{row['commanded']}' at {t_cmd:.6f} → match at {t_match:.6f} "
                  f"→ latency {lat_ms:.3f} ms")
    else:
        print("[Info] No command changes ≥ min latency threshold.")

    # Save filtered latency table (seconds)
    if OUT_CSV:
        lat_df.to_csv(OUT_CSV, index=False)
        print(f"[Saved] {OUT_CSV}")

    # Plot histogram + normal-fit in **milliseconds** (filtered)
    plot_latency_histogram_ms(lat_df, out_png=OUT_PNG)


if __name__ == "__main__":
    main()
