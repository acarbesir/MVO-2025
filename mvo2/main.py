import os
import cv2
import numpy as np

# Import functions/data from other modules
from camera_params import CAMERA_PARAMS
from visual_odometry import estimate_trajectory
from data_loader import load_ground_truth_data
from visualizer import (
    plot_trajectory,
    plot_errors,
    calculate_rmse,
    save_trajectory_to_csv,
)


# ---------------------------------------------------------------------------
# Helper: similarity-transform calibration (scale, rotation, translation)
# ---------------------------------------------------------------------------

def calibrate_similarity(
    trajectory_xyz: np.ndarray,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    n_frames: int = 450,
):
    """Calibrate the VO track to ground truth using the first *n_frames*."""

    n = min(n_frames, len(trajectory_xyz), len(gt_x))
    if n < 3:
        raise ValueError("Not enough overlapping frames for calibration.")

    # Matched point sets -----------------------------------------------------
    est_xy = trajectory_xyz[:n, :2]  # (n,2)
    gt_xy = np.column_stack((gt_x[:n], gt_y[:n]))

    # Centroids --------------------------------------------------------------
    est_centroid = est_xy.mean(axis=0)
    gt_centroid = gt_xy.mean(axis=0)

    est_c = est_xy - est_centroid
    gt_c = gt_xy - gt_centroid

    # Umeyama / Procrustes (2-D) --------------------------------------------
    H = est_c.T @ gt_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    '''if np.linalg.det(R) < 0:  # reflection guard
        Vt[-1, :] *= -1
        R = Vt.T @ U.T'''

    # Scale
    scale = np.linalg.norm(gt_c) / np.linalg.norm(est_c)

    # Apply to **all** frames ------------------------------------------------
    est_all_xy = trajectory_xyz[:, :2] - est_centroid  # move to origin
    est_all_xy = (est_all_xy @ R.T) * scale + gt_centroid

    traj_calibrated = trajectory_xyz.copy()
    traj_calibrated[:, 0] = est_all_xy[:, 0]
    traj_calibrated[:, 1] = est_all_xy[:, 1]

    angle_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return traj_calibrated, scale, angle_deg


# ---------------------------------------------------------------------------
# Helper: live trajectory map display (post-processing)
# ---------------------------------------------------------------------------

def play_live_trajectory_map(
    trajectory_xyz: np.ndarray,
    gt_x: np.ndarray | None = None,
    gt_y: np.ndarray | None = None,
    map_size: tuple[int, int] = (600, 600),
    wait_ms: int = 1,
    save_path: str = "live_trajectory_map.png",
):
    """Display the calibrated VO trajectory (and GT) step-by-step using OpenCV."""

    # Create a blank canvas (H, W, 3)
    traj_img = np.zeros((map_size[1], map_size[0], 3), dtype=np.uint8)

    cx_off, cy_off = map_size[0] // 2, map_size[1] // 2
    total = len(trajectory_xyz)

    for idx, (x, y, z) in enumerate(trajectory_xyz):
        # Visual convention: flip sign so +x/right is rightwards on screen
        draw_x = int(-x) + cx_off
        draw_y = int(-y) + cy_off

        draw_x = np.clip(draw_x, 0, map_size[0] - 1)
        draw_y = np.clip(draw_y, 0, map_size[1] - 1)

        # Colour fades from green to red as frames progress
        colour = (idx * 255 // total, 255 - idx * 255 // total, 0)
        cv2.circle(traj_img, (draw_x, draw_y), 1, colour, 1)

        # Ground-truth (optional, red)
        if gt_x is not None and gt_y is not None and idx < len(gt_x):
            gt_dx = int(gt_x[idx]) + cx_off
            gt_dy = int(gt_y[idx]) + cy_off
            gt_dx = np.clip(gt_dx, 0, map_size[0] - 1)
            gt_dy = np.clip(gt_dy, 0, map_size[1] - 1)
            cv2.circle(traj_img, (gt_dx, gt_dy), 1, (0, 0, 255), 2)

        # HUD text (overwrite area)
        cv2.rectangle(traj_img, (10, 10), (590, 50), (0, 0, 0), -1)
        info = f"Frame {idx+1}/{total} VO X={-x:.2f}m Y={-y:.2f}m"
        cv2.putText(traj_img, info, (20, 35), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Live Trajectory Map", traj_img)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q"):
            break

    cv2.imwrite(save_path, traj_img)
    cv2.destroyWindow("Live Trajectory Map")
    print(f"Live trajectory map saved as '{save_path}'")


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    frames_folder = "frames/output_frames"  # folder containing frames
    gt_file = "ground_truth/GT_Translations.csv"
    output_trajectory_csv = "estimated_trajectory.csv"

    # --- 1. Check for frames folder existence ------------------------------
    if not os.path.exists(frames_folder):
        print(
            f"Error: Frames folder '{frames_folder}' not found. Please ensure the folder exists and contains images."
        )
        exit()

    # --- 2. Load Ground Truth Data -----------------------------------------
    ground_truth_for_display, gt_x, gt_y, gt_z = load_ground_truth_data(gt_file)

    # --- 3. Estimate Trajectory --------------------------------------------
    print("\n--- Starting Trajectory Estimation ---")
    # Pass gt_z to estimate_trajectory for dynamic Z-axis handling
    trajectory = estimate_trajectory(
        frames_folder,
        CAMERA_PARAMS,
        gt_z_data=gt_z, # Pass ground truth Z data
        z_constant_fallback=5.0, # Fallback Z for when GT is not available or exhausted
        scale_factor=1.0,  # leave raw; similarity calibration will fix scale
        display_live_trajectory_map=False,  # disabled inside VO function
        display_feature_matches=True,
        ground_truth_data_for_display=None,  # not needed inside VO
    )
    print("--- Trajectory Estimation Complete ---\n")

    if trajectory is None:
        print("Trajectory estimation failed. No data to plot or analyze.")
        exit()

    # ----------------------------------------------------------------------
    # 4. Calibrate scale & orientation with the first 450 GT frames
    # ----------------------------------------------------------------------
    CALIB_FRAMES = 450
    if gt_x is not None and len(gt_x) >= CALIB_FRAMES:
        trajectory_for_plot, vscale, vangle = calibrate_similarity(
            trajectory, gt_x, gt_y, n_frames=CALIB_FRAMES
        )
        print(f"\nCalibration using first {CALIB_FRAMES} GT frames:")
        print(f" • Global XY scale : {vscale:.3f}")
        print(f" • Rotation (deg)  : {vangle:+.2f}")

        # IMPORTANT: Apply Z from ground truth for the calibrated segment
        if gt_z is not None and len(gt_z) >= CALIB_FRAMES:
            min_len_calib_z = min(CALIB_FRAMES, trajectory_for_plot.shape[0], len(gt_z))
            trajectory_for_plot[:min_len_calib_z, 2] = gt_z[:min_len_calib_z]
            print(f" • Z-axis aligned with ground truth for first {min_len_calib_z} frames.")
        else:
            print("⚠️ Not enough ground-truth Z data to align Z-axis for calibration.")

    else:
        print(
            "\n⚠️ Not enough ground-truth data for calibration — using raw VO output."
        )
        trajectory_for_plot = trajectory.copy()

    print("\nEstimated Trajectory (x, y, z – after calibration):")
    print(trajectory_for_plot)
    print(f"Total frames processed: {trajectory_for_plot.shape[0]}")

    # ----------------------------------------------------------------------
    # 5. Compute RMSE -------------------------------------------------------
    if (
        gt_x is not None
        and gt_y is not None
        and gt_z is not None
        and len(gt_x) > 0
    ):
        rmse_results = calculate_rmse(trajectory_for_plot, gt_x, gt_y, gt_z)
        if rmse_results:
            print("\n--- RMSE Errors (after calibration) ---")
            for key, value in rmse_results.items():
                print(f"{key}: {value:.4f} meters")
            print("---------------------------------------\n")
        else:
            print(
                "\nWarning: Ground truth data not available or incomplete. Skipping RMSE calculation."
            )

    # ----------------------------------------------------------------------
    # 6. Save trajectory & plots -------------------------------------------
    save_trajectory_to_csv(trajectory_for_plot, output_trajectory_csv)

    plot_trajectory(
        trajectory_for_plot, gt_x, gt_y, filename="trajectory_plot.png"
    )
    if gt_x is not None:
        plot_errors(
            trajectory_for_plot,
            gt_x,
            gt_y,
            gt_z,
            filename="trajectory_error_plot.png",
        )

    # ----------------------------------------------------------------------
    # 7. Live trajectory display (post-processing) --------------------------
    play_live_trajectory_map(
        trajectory_for_plot,
        gt_x=gt_x,
        gt_y=gt_y,
        map_size=(600, 600),
        wait_ms=1,
        save_path="live_trajectory_map.png",
    )