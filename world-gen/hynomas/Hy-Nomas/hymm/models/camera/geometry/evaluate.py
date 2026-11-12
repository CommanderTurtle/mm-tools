"""
Evaluate and visualize camera action discretization results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from discretize_poses import (
    discretize_poses_worldplay,
    reconstruct_trajectory_worldplay,
)


def compute_trajectory_metrics(original_c2ws, reconstructed_c2ws, align_scale=True):
    """
    Compute trajectory evaluation metrics including translation and rotation errors.

    Args:
        original_c2ws: np.ndarray [T, 4, 4] - original camera-to-world matrices
        reconstructed_c2ws: np.ndarray [T, 4, 4] - reconstructed camera-to-world matrices
        align_scale: bool - whether to align scale between trajectories

    Returns:
        metrics: dict containing evaluation metrics
    """
    assert original_c2ws.shape == reconstructed_c2ws.shape, (
        f"Shape mismatch: {original_c2ws.shape} vs {reconstructed_c2ws.shape}"
    )

    num_frames = len(original_c2ws)

    # Extract positions and rotations
    orig_positions = original_c2ws[:, :3, 3]
    recon_positions = reconstructed_c2ws[:, :3, 3]
    orig_rotations = original_c2ws[:, :3, :3]
    recon_rotations = reconstructed_c2ws[:, :3, :3]

    # ==================== Translation Metrics ====================

    # Center trajectories (align starting point)
    orig_centered = orig_positions - orig_positions[0]
    recon_centered = recon_positions - recon_positions[0]

    # Compute scale factor if needed
    orig_length = np.sum(np.linalg.norm(np.diff(orig_centered, axis=0), axis=1))
    recon_length = np.sum(np.linalg.norm(np.diff(recon_centered, axis=0), axis=1))

    if align_scale and recon_length > 1e-6:
        scale_factor = orig_length / recon_length
        recon_scaled = recon_centered * scale_factor
    else:
        scale_factor = 1.0
        recon_scaled = recon_centered

    # ATE (Absolute Trajectory Error) - RMSE of position errors
    ate_per_frame = np.linalg.norm(orig_centered - recon_scaled, axis=1)
    ate_rmse = np.sqrt(np.mean(ate_per_frame**2))

    # RPE Translation (Relative Pose Error) - RMSE of relative position errors
    orig_deltas = np.diff(orig_centered, axis=0)
    recon_deltas = np.diff(recon_scaled, axis=0)
    rpe_trans_per_frame = np.linalg.norm(orig_deltas - recon_deltas, axis=1)
    rpe_trans_rmse = np.sqrt(np.mean(rpe_trans_per_frame**2))

    # ==================== Rotation Metrics ====================

    # Compute relative rotation from first frame and rotation error
    rotation_errors_deg = []
    for i in range(num_frames):
        orig_rot_rel = orig_rotations[0].T @ orig_rotations[i]
        recon_rot_rel = recon_rotations[0].T @ recon_rotations[i]
        R_error = orig_rot_rel.T @ recon_rot_rel
        trace = np.trace(R_error)
        cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
        rotation_errors_deg.append(np.degrees(np.arccos(cos_angle)))

    rotation_errors_deg = np.array(rotation_errors_deg)
    rot_error_mean = np.mean(rotation_errors_deg)

    # RPE Rotation - RMSE of relative rotation errors
    rpe_rot_errors_deg = []
    for i in range(1, num_frames):
        orig_dR = orig_rotations[i - 1].T @ orig_rotations[i]
        recon_dR = recon_rotations[i - 1].T @ recon_rotations[i]
        R_error = orig_dR.T @ recon_dR
        trace = np.trace(R_error)
        cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
        rpe_rot_errors_deg.append(np.degrees(np.arccos(cos_angle)))

    rpe_rot_rmse = (
        np.sqrt(np.mean(np.array(rpe_rot_errors_deg) ** 2))
        if rpe_rot_errors_deg
        else 0.0
    )

    # ==================== Compile Metrics ====================

    metrics = {
        # Translation
        "ate_rmse": ate_rmse,  # Absolute Trajectory Error (位置误差)
        "rpe_trans_rmse": rpe_trans_rmse,  # Relative Pose Error - Translation (相对平移误差)
        # Rotation (degrees)
        "rot_error_mean": rot_error_mean,  # Mean rotation error (平均旋转误差)
        "rpe_rot_rmse": rpe_rot_rmse,  # Relative Pose Error - Rotation (相对旋转误差)
    }

    return metrics


def print_metrics(metrics, title="Trajectory Metrics"):
    """Print evaluation metrics in a formatted way."""
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print("=" * 50)
    print(f"  ATE RMSE (trans):     {metrics['ate_rmse']:.6f}")
    print(f"  RPE RMSE (trans):     {metrics['rpe_trans_rmse']:.6f}")
    print(f"  Rot Error Mean:       {metrics['rot_error_mean']:.4f}°")
    print(f"  RPE RMSE (rot):       {metrics['rpe_rot_rmse']:.4f}°")


def visualize_trajectory_comparison(
    original_positions, reconstructed_positions, title="Trajectory Comparison"
):
    """
    Visualize comparison between original trajectory and reconstructed trajectory.

    Args:
        original_positions: np.ndarray [T, 3] - original camera positions
        reconstructed_positions: np.ndarray [T, 3] - reconstructed camera positions
        title: str - figure title

    Returns:
        fig: matplotlib figure
    """
    fig = plt.figure(figsize=(18, 6))

    # Helper function to set consistent axis limits
    def set_axis_limits(ax, positions_list):
        all_positions = np.vstack(positions_list)
        margin = 0.1
        for i, label in enumerate(["X", "Y", "Z"]):
            min_val, max_val = all_positions[:, i].min(), all_positions[:, i].max()
            range_val = max_val - min_val
            if range_val < 1e-6:
                range_val = 1.0
            if label == "X":
                ax.set_xlim(min_val - margin * range_val, max_val + margin * range_val)
            elif label == "Y":
                ax.set_ylim(min_val - margin * range_val, max_val + margin * range_val)
            else:
                ax.set_zlim(min_val - margin * range_val, max_val + margin * range_val)

    # Plot 1: Original trajectory
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot(
        original_positions[:, 0],
        original_positions[:, 1],
        original_positions[:, 2],
        "b-",
        linewidth=2,
        alpha=0.8,
        label="Original",
    )
    ax1.scatter(
        original_positions[0, 0],
        original_positions[0, 1],
        original_positions[0, 2],
        c="green",
        s=100,
        marker="o",
        label="Start",
    )
    ax1.scatter(
        original_positions[-1, 0],
        original_positions[-1, 1],
        original_positions[-1, 2],
        c="red",
        s=100,
        marker="s",
        label="End",
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Original Trajectory\n(Before Discretization)")
    ax1.legend(loc="upper left", fontsize=8)
    set_axis_limits(ax1, [original_positions])

    # Plot 2: Reconstructed trajectory
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.plot(
        reconstructed_positions[:, 0],
        reconstructed_positions[:, 1],
        reconstructed_positions[:, 2],
        "r-",
        linewidth=2,
        alpha=0.8,
        label="Reconstructed",
    )
    ax2.scatter(
        reconstructed_positions[0, 0],
        reconstructed_positions[0, 1],
        reconstructed_positions[0, 2],
        c="green",
        s=100,
        marker="o",
        label="Start",
    )
    ax2.scatter(
        reconstructed_positions[-1, 0],
        reconstructed_positions[-1, 1],
        reconstructed_positions[-1, 2],
        c="red",
        s=100,
        marker="s",
        label="End",
    )
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Reconstructed Trajectory\n(After Discretization)")
    ax2.legend(loc="upper left", fontsize=8)
    set_axis_limits(ax2, [reconstructed_positions])

    # Plot 3: Comparison (both overlaid)
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot(
        original_positions[:, 0],
        original_positions[:, 1],
        original_positions[:, 2],
        "b-",
        linewidth=2,
        alpha=0.7,
        label="Original",
    )
    ax3.plot(
        reconstructed_positions[:, 0],
        reconstructed_positions[:, 1],
        reconstructed_positions[:, 2],
        "r--",
        linewidth=2,
        alpha=0.7,
        label="Reconstructed",
    )
    ax3.scatter(
        original_positions[0, 0],
        original_positions[0, 1],
        original_positions[0, 2],
        c="green",
        s=100,
        marker="o",
        label="Start",
    )
    ax3.scatter(
        original_positions[-1, 0],
        original_positions[-1, 1],
        original_positions[-1, 2],
        c="blue",
        s=80,
        marker="s",
        label="Original End",
    )
    ax3.scatter(
        reconstructed_positions[-1, 0],
        reconstructed_positions[-1, 1],
        reconstructed_positions[-1, 2],
        c="red",
        s=80,
        marker="^",
        label="Recon End",
    )
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("Comparison\n(Blue=Original, Red=Reconstructed)")
    ax3.legend(loc="upper left", fontsize=8)
    set_axis_limits(ax3, [original_positions, reconstructed_positions])

    # Compute and display error metrics
    # Normalize positions for fair comparison
    orig_centered = original_positions - original_positions[0]
    recon_centered = reconstructed_positions - reconstructed_positions[0]

    # Scale reconstructed to match original trajectory length
    orig_length = np.sum(np.linalg.norm(np.diff(orig_centered, axis=0), axis=1))
    recon_length = np.sum(np.linalg.norm(np.diff(recon_centered, axis=0), axis=1))
    if recon_length > 1e-6:
        scale_factor = orig_length / recon_length
        recon_scaled = recon_centered * scale_factor
    else:
        recon_scaled = recon_centered
        scale_factor = 1.0

    # Compute position error
    position_errors = np.linalg.norm(orig_centered - recon_scaled, axis=1)
    mean_error = np.mean(position_errors)
    max_error = np.max(position_errors)

    plt.suptitle(
        f"{title}\nMean Position Error: {mean_error:.4f}, Max Error: {max_error:.4f}, Scale: {scale_factor:.2f}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


def evaluate_trajectory(json_path, output_dir=None, tps=False):
    """
    Load trajectory from JSON, discretize poses, reconstruct trajectory, compute metrics, and visualize.

    Args:
        json_path: path to pose JSON file
        output_dir: directory to save figures (if None, just display)
        tps: whether to use TPS mode for discretization

    Returns:
        w2cs, intrinsics, action_labels, metrics
    """
    json_path = Path(json_path)

    # Load pose data
    with open(json_path, "r") as f:
        pose_dict = json.load(f)

    print(f"Loaded {len(pose_dict)} frames from {json_path.name}")

    # Discretize poses
    w2cs, intrinsics, action_labels = discretize_poses_worldplay(pose_dict, tps=tps)

    print(f"W2C shape: {w2cs.shape}")
    print(f"Intrinsics shape: {intrinsics.shape}")
    print(f"Action labels shape: {action_labels.shape}")

    # Get original c2ws and positions
    pose_keys = sorted(pose_dict.keys(), key=lambda x: int(x))
    original_c2ws = np.stack([np.array(pose_dict[k]["extrinsic"]) for k in pose_keys])
    original_positions = original_c2ws[:, :3, 3]

    # Get initial c2w for reconstruction
    initial_c2w = original_c2ws[0]

    # Estimate translation scale from original trajectory
    if len(original_positions) > 1:
        displacements = np.linalg.norm(np.diff(original_positions, axis=0), axis=1)
        trans_scale = (
            np.median(displacements[displacements > 1e-6])
            if np.any(displacements > 1e-6)
            else 0.08
        )
    else:
        trans_scale = 0.08

    # Reconstruct trajectory from action labels
    reconstructed_c2ws, reconstructed_positions = reconstruct_trajectory_worldplay(
        action_labels,
        initial_c2w=initial_c2w,
        trans_scale=trans_scale,
    )

    # Compute evaluation metrics
    metrics = compute_trajectory_metrics(
        original_c2ws, reconstructed_c2ws, align_scale=True
    )
    print_metrics(metrics, title=f"Metrics for {json_path.name}")

    # Visualize
    title = json_path.stem
    fig = visualize_trajectory_comparison(
        original_positions,
        reconstructed_positions,
        title=f"{title} - Discretization Comparison",
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"{title}_comparison.png", dpi=150, bbox_inches="tight"
        )

        # Save metrics to JSON
        metrics_to_save = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items()
        }
        with open(output_dir / f"{title}_metrics.json", "w") as f:
            json.dump(metrics_to_save, f, indent=2)

        print(f"\nFigures and metrics saved to {output_dir}")
    else:
        plt.show()

    plt.close("all")

    return w2cs, intrinsics, action_labels, metrics


def main():
    """Process all trajectory files in the trajectories folder."""
    traj_dir = Path(__file__).parent / "trajectories"
    output_dir = Path(__file__).parent / "visualization_results"

    json_files = list(traj_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {traj_dir}")
        return

    print(f"Found {len(json_files)} trajectory files")

    for json_path in json_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {json_path.name}")
        print("=" * 60)
        evaluate_trajectory(json_path, output_dir=output_dir)


if __name__ == "__main__":
    main()
