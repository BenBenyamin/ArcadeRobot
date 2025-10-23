import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# -------------------------------
# USER SETTINGS
# -------------------------------

LOGDIR_PARENT = "./PONG_tensorboard"
TAG = "rollout/ep_rew_mean"
SAVE_PATH = "ep_rew_mean_comparison.png"

RUN_LABELS = {
    "PPO_Delay=30_Steps=100.0M_1": "PPO with 30 frame stack",
    "RPPO_Delay=30_Stack=4_1": "Recurrent PPO with 4 frame stack",
    "RPPO_Delay=30_Steps=100.0M_1": "Recurrent PPO (no frame stack)",
}


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def load_scalar(logdir, tag, max_points=None):
    """Load one scalar tag from a TensorBoard log directory."""
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={event_accumulator.SCALARS: 0},  # load only scalars
    )
    ea.Reload()

    available = ea.Tags().get("scalars", [])
    if tag not in available:
        print(f"Tag '{tag}' not found in {logdir}. Available: {available}")
        return None, None

    scalars = ea.Scalars(tag)

    # Optionally limit to first N points
    if max_points:
        scalars = scalars[:max_points]

    steps = [s.step for s in scalars]
    values = [s.value for s in scalars]
    return steps, values


def plot_multiple_runs(parent_dir, tag, run_labels, save=None):
    """Plot full and zoomed-in versions of a scalar across multiple runs."""
    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(9, 10))
    runs_plotted = 0

    for subdir in sorted(os.listdir(parent_dir)):
        path = os.path.join(parent_dir, subdir)
        if not os.path.isdir(path):
            continue

        steps, values = load_scalar(path, tag)
        if steps is None or values is None:
            continue

        label = run_labels.get(subdir, subdir)
        ax_full.plot(steps, values, label=label)
        ax_zoom.plot(steps[:100], values[:100], label=label)
        runs_plotted += 1

    if runs_plotted == 0:
        print("No valid runs found with that tag.")
        return

    # Full range plot
    ax_full.set_xlabel("Steps")
    ax_full.set_ylabel(tag)
    ax_full.set_title(f"{tag} (Full Range)")
    ax_full.grid(True, alpha=0.3)
    ax_full.legend()

    # Zoomed subplot (first 100 steps)
    ax_zoom.set_xlabel("Steps (first 100)")
    ax_zoom.set_ylabel(tag)
    ax_zoom.set_title(f"{tag} (First 100 Steps)")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend()

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300)
        print(f"Saved plot to {save}")
    else:
        plt.show()



plot_multiple_runs(LOGDIR_PARENT, TAG, RUN_LABELS, save=SAVE_PATH)
