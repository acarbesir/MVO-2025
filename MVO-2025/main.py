
import os
import numpy as np

# Import functions/data from other modules
from camera_params import CAMERA_PARAMS
from visual_odometry import estimate_trajectory # Commented out, using the modified one directly
from data_loader import load_ground_truth_data # Still imported but its function is a no-op
from visualizer import plot_trajectory # Now expects a 3D plot

if __name__ == "__main__":
    video_file = "Ornek_Veri_Gunduz_Kamera_VO.MP4"
    gt_file = "GT_Translations.csv" # Still defined but not used

    # --- 1. Check for video file existence ---
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found. Please ensure the video is in the same directory or provide the full path.")
        exit()

    # --- 2. Load Ground Truth Data (Skipped) ---
    # The function will return None values, effectively disabling GT usage.
    # ground_truth_for_display, gt_x, gt_y, gt_z = load_ground_truth_data(gt_file)
    # No need to call this if not using it for display.

    # --- 3. Estimate Trajectory ---
    print("\n--- Starting Trajectory Estimation ---")
    trajectory = estimate_trajectory(
        video_file,
        CAMERA_PARAMS,
        scale_factor=7.0,      # Tunable parameter for displacement
        display_live_trajectory_map=True,
        display_feature_matches=True
    )
    print("--- Trajectory Estimation Complete ---\n")

    if trajectory is not None:
        # Create a copy for plotting and alignment to avoid modifying the original
        trajectory_for_plot = trajectory.copy()

        # The previous code had `display_x_est = -current_x_world` and `display_y_est = -current_y_world`
        # for 2D visualization. If your desired world coordinate system has positive X to the right
        # and positive Y upwards (or North), and Z upwards, you might need to adjust these signs.
        # For now, let's keep the previous flips for X and Y for consistency with the 2D plot,
        # but the Z will be as estimated (which is currently a simplistic accumulation).
        trajectory_for_plot[:, 0] = -trajectory_for_plot[:, 0] # Flip X for plotting consistency
        trajectory_for_plot[:, 1] = -trajectory_for_plot[:, 1] # Flip Y for plotting consistency
        # Z-axis is now constant at 10.0 for all points, no further adjustment needed here

        print("\nEstimated Trajectory (x, y, z - for plotting):")
        print(trajectory_for_plot)
        print(f"Total frames processed: {trajectory_for_plot.shape[0]}")

        # --- 4. Plot Trajectories ---
        plot_trajectory(trajectory_for_plot) # Now plots in 3D
    else:
        print("Trajectory estimation failed. No data to plot or analyze.")