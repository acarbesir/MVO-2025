# visualizer.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_trajectory(estimated_trajectory, ground_truth_x=None, ground_truth_y=None, filename="trajectory.png"):
    """
    Plots the estimated and (optionally) ground truth 2D trajectories.

    Args:
        estimated_trajectory (numpy.ndarray): N x 3 array of (x, y, z) estimated trajectory.
        ground_truth_x (numpy.ndarray, optional): Array of ground truth X positions.
        ground_truth_y (numpy.ndarray, optional): Array of ground truth Y positions.
        filename (str): Name of the file to save the plot.
    """
    try:
        plt.figure(figsize=(10, 8))
        plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], label="Estimated Trajectory")

        if ground_truth_x is not None and ground_truth_y is not None:
            plt.plot(ground_truth_x, ground_truth_y, label="Ground Truth Trajectory", linestyle='--', color='red')

        plt.xlabel("X Position (meters)")
        plt.ylabel("Y Position (meters)")
        plt.title("Estimated vs. Ground Truth Trajectory (2D)")
        plt.grid(True)
        plt.axis('equal') # Ensures that 1 unit on x-axis is equal to 1 unit on y-axis
        plt.legend()
        plt.savefig(filename)
        plt.show()
    except ImportError:
        print("\nMatplotlib not installed. Skipping trajectory plot.")
        print("To install: pip install matplotlib")
    except Exception as e:
        print(f"Error while plotting trajectory: {e}")

def plot_errors(estimated_trajectory, ground_truth_x, ground_truth_y, ground_truth_z, filename="trajectory_error.png"):
    """
    Plots the error values between estimated and ground truth trajectories over frames.

    Args:
        estimated_trajectory (numpy.ndarray): N x 3 array of (x, y, z) estimated trajectory.
        ground_truth_x (numpy.ndarray): Array of ground truth X positions.
        ground_truth_y (numpy.ndarray): Array of ground truth Y positions.
        ground_truth_z (numpy.ndarray): Array of ground truth Z positions.
        filename (str): Name of the file to save the plot.
    """
    if ground_truth_x is None or ground_truth_y is None or ground_truth_z is None:
        print("Cannot plot error values: Ground truth data is incomplete.")
        return

    try:
        min_len = min(len(estimated_trajectory), len(ground_truth_x))
        estimated_traj_aligned = estimated_trajectory[:min_len]
        gt_x_aligned = ground_truth_x[:min_len]
        gt_y_aligned = ground_truth_y[:min_len]
        gt_z_aligned = ground_truth_z[:min_len]

        error_x = estimated_traj_aligned[:, 0] - gt_x_aligned
        error_y = estimated_traj_aligned[:, 1] - gt_y_aligned
        error_z = estimated_traj_aligned[:, 2] - gt_z_aligned

        frames = np.arange(min_len)

        plt.figure(figsize=(12, 7))
        plt.plot(frames, error_x, color='red', label='Error X [m]')
        plt.plot(frames, error_y, color='green', label='Error Y [m]')
        plt.plot(frames, error_z, color='blue', label='Error Z [m]')

        plt.xlabel("Frame Number")
        plt.ylabel("Position Deviation [m]")
        plt.title("Error Values Calculated Using Reference and Estimated Position Information")
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        plt.show()
    except ImportError:
        print("\nMatplotlib not installed. Skipping error plot.")
        print("To install: pip install matplotlib")
    except Exception as e:
        print(f"Error while plotting errors: {e}")

def calculate_rmse(estimated_trajectory, ground_truth_x, ground_truth_y, ground_truth_z):
    """
    Calculates the Root Mean Squared Error (RMSE) between estimated and ground truth trajectories.

    Args:
        estimated_trajectory (numpy.ndarray): N x 3 array of (x, y, z) estimated trajectory.
        ground_truth_x (numpy.ndarray): Array of ground truth X positions.
        ground_truth_y (numpy.ndarray): Array of ground truth Y positions.
        ground_truth_z (numpy.ndarray): Array of ground truth Z positions.

    Returns:
        dict: A dictionary containing RMSE values for X, Y, Z, and combined 3D.
              Returns None if ground truth data is incomplete.
    """
    if ground_truth_x is None or ground_truth_y is None or ground_truth_z is None:
        print("Cannot calculate RMSE: Ground truth data is incomplete.")
        return None

    min_len = min(len(estimated_trajectory), len(ground_truth_x))
    estimated_traj_aligned = estimated_trajectory[:min_len]
    gt_x_aligned = ground_truth_x[:min_len]
    gt_y_aligned = ground_truth_y[:min_len]
    gt_z_aligned = ground_truth_z[:min_len]

    error_x = estimated_traj_aligned[:, 0] - gt_x_aligned
    error_y = estimated_traj_aligned[:, 1] - gt_y_aligned
    error_z = estimated_traj_aligned[:, 2] - gt_z_aligned

    rmse_x = np.sqrt(np.mean(error_x**2))
    rmse_y = np.sqrt(np.mean(error_y**2))
    rmse_z = np.sqrt(np.mean(error_z**2))
    rmse_3d = np.sqrt(np.mean(error_x**2 + error_y**2 + error_z**2))

    return {
        "RMSE X": rmse_x,
        "RMSE Y": rmse_y,
        "RMSE Z": rmse_z,
        "RMSE 3D (Combined)": rmse_3d
    }

def save_trajectory_to_csv(trajectory_data, filename="estimated_trajectory.csv"):
    """
    Saves the estimated trajectory data to a CSV file.

    Args:
        trajectory_data (numpy.ndarray): N x 3 array of (x, y, z) trajectory.
        filename (str): Name of the CSV file to save the data.
    """
    try:
        df = pd.DataFrame(trajectory_data, columns=['translation_x', 'translation_y', 'translation_z'])
        df.to_csv(filename, index=False)
        print(f"\nEstimated trajectory saved to '{filename}' successfully.")
    except ImportError:
        print("\nPandas not installed. Skipping CSV export.")
        print("To install: pip install pandas")
    except Exception as e:
        print(f"Error while saving trajectory to CSV: {e}")
