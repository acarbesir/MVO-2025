
import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(estimated_trajectory, filename="trajectory_3d.png"):
    """
    Plots the estimated 3D trajectory.

    Args:
        estimated_trajectory (numpy.ndarray): N x 3 array of (x, y, z) estimated trajectory.
        filename (str): Name of the file to save the plot.
    """
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], estimated_trajectory[:, 2], label="Estimated Trajectory")

        ax.set_xlabel("X Position (meters)")
        ax.set_ylabel("Y Position (meters)")
        ax.set_zlabel("Z Position (meters)")
        ax.set_title("Estimated 3D Trajectory")
        ax.grid(True)
        ax.legend()
        plt.savefig(filename)
        plt.show()
    except ImportError:
        print("\nMatplotlib not installed. Skipping trajectory plot.")
        print("To install: pip install matplotlib")
    except Exception as e:
        print(f"Error while plotting trajectory: {e}")

def plot_errors(estimated_trajectory, ground_truth_x, ground_truth_y, ground_truth_z, filename="trajectory_error.png"):
    """
    This function is no longer needed as ground truth data is not being used.
    """
    print("Error plotting is skipped as ground truth data is not available.")
    return

def calculate_rmse(estimated_trajectory, ground_truth_x, ground_truth_y, ground_truth_z):
    """
    This function is no longer needed as ground truth data is not being used.
    """
    print("RMSE calculation is skipped as ground truth data is not available.")
    return None

