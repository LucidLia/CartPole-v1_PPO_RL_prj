import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt


RUN_DIR = "runs"
OUT_ALL = os.path.join(RUN_DIR, "plot_learning_curves_all.png")
OUT_PPO = os.path.join(RUN_DIR, "plot_learning_curves_ppo_clips.png")


def moving_average(x, window=20):
    if len(x) == 0:
        return np.array([])
    w = min(window, len(x))
    out = []
    for i in range(len(x)):
        s = max(0, i - w + 1)
        out.append(np.mean(x[s:i + 1]))
    return np.array(out, dtype=np.float32)


def parse_curve_filename(path):
    """
    Expected filenames:
      - curves_REINFORCE_seed0.npz
      - curves_PPO_seed1_clip0.1.npz
    """
    name = os.path.basename(path)

    if name.startswith("curves_REINFORCE"):
        m = re.search(r"seed(\d+)", name)
        seed = int(m.group(1)) if m else None
        return ("REINFORCE", None, seed)

    if name.startswith("curves_PPO"):
        m_seed = re.search(r"seed(\d+)", name)

        # ✅ safer clip regex: capture only numeric like 0.1 or 0.2
        # and stop before extension
        m_clip = re.search(r"clip([0-9]+(?:\.[0-9]+)?)", name)

        seed = int(m_seed.group(1)) if m_seed else None
        clip = float(m_clip.group(1)) if m_clip else None

        return ("PPO", clip, seed)

    return (None, None, None)



def load_grouped_curves():
    files = sorted(glob.glob(os.path.join(RUN_DIR, "curves_*.npz")))
    if not files:
        raise FileNotFoundError(f"No curve files found in '{RUN_DIR}/'.")

    groups = {}  # key: (algo, clip) -> list of returns arrays

    for f in files:
        algo, clip, seed = parse_curve_filename(f)
        if algo is None:
            continue

        data = np.load(f)
        returns = data["returns"].astype(np.float32)

        key = (algo, clip)
        groups.setdefault(key, []).append(returns)

    return groups


def stack_by_min_length(curves):
    """
    Curves can have different lengths.
    We align by the minimum length to keep comparison fair.
    """
    min_len = min(len(c) for c in curves)
    trimmed = [c[:min_len] for c in curves]
    stacked = np.stack(trimmed, axis=0)  # [num_seeds, T]
    return stacked


def plot_group_mean_std(groups, only_ppo=False):
    """
    Draw mean ± std curves.
    - only_ppo=True -> include only PPO keys
    """
    keys = sorted(groups.keys(), key=lambda k: (k[0], 999 if k[1] is None else k[1]))

    plt.figure()

    any_plotted = False

    for (algo, clip) in keys:
        if only_ppo and algo != "PPO":
            continue

        curves = groups[(algo, clip)]
        if len(curves) == 0:
            continue

        stacked = stack_by_min_length(curves)
        # Use moving average to smooth each seed curve before aggregation
        smoothed = np.stack([moving_average(c, 20) for c in stacked], axis=0)

        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)

        if algo == "REINFORCE":
            label = "REINFORCE (mean ± std)"
        else:
            label = f"PPO clip={clip} (mean ± std)"

        x = np.arange(len(mean))
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

        any_plotted = True

    if not any_plotted:
        raise RuntimeError("No curves were plotted. Check file naming or contents.")

    plt.xlabel("Episode index (aligned by min length)")
    plt.ylabel("Return (moving avg window=20)")
    title = "Learning Curves Comparison (mean ± std over seeds)"
    if only_ppo:
        title = "PPO Clip Ablation Learning Curves (mean ± std over seeds)"
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def main():
    os.makedirs(RUN_DIR, exist_ok=True)

    groups = load_grouped_curves()

    # 1) All (REINFORCE + PPO clips)
    plot_group_mean_std(groups, only_ppo=False)
    plt.savefig(OUT_ALL, dpi=200)
    plt.close()
    print(f"Saved: {OUT_ALL}")

    # 2) PPO-only clip comparison
    plot_group_mean_std(groups, only_ppo=True)
    plt.savefig(OUT_PPO, dpi=200)
    plt.close()
    print(f"Saved: {OUT_PPO}")

    # Quick console summary
    print("\n[Group summary]")
    for k, curves in sorted(groups.items(), key=lambda x: (x[0][0], 999 if x[0][1] is None else x[0][1])):
        algo, clip = k
        lens = [len(c) for c in curves]
        print(f" - {algo} clip={clip}: {len(curves)} seeds, lengths={lens}")


if __name__ == "__main__":
    main()
