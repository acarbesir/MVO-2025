# data_loader.py (Updated to load Z-axis ground truth)

import pandas as pd
import numpy as np
import os

def load_ground_truth_data(gt_file_path):
    """
    Loads ground truth trajectory data from a CSV file.

    Args:
    gt_file_path (str): Path to the ground truth CSV file.
    Expected columns: 'translation_x', 'translation_y', 'translation_z'.

    Returns:
    tuple: (ground_truth_data (numpy.ndarray), gt_x (numpy.ndarray),
            gt_y (numpy.ndarray), gt_z (numpy.ndarray)).
            Returns (None, None, None, None) if file not found or error.
    """
    if not os.path.exists(gt_file_path):
        print(f"Warning: Ground truth file '{gt_file_path}' not found.")
        return None, None, None, None

    print(f"\nLoading ground truth from {gt_file_path}...")
    try:
        ground_truth_df = pd.read_csv(gt_file_path)

        # Check if 'translation_z' column exists
        if 'translation_z' not in ground_truth_df.columns:
            print(f"Warning: 'translation_z' column not found in '{gt_file_path}'. Assuming Z=0 or using default.")
            gt_z = np.zeros_like(ground_truth_df['translation_x'].values) # Fallback to zeros
        else:
            gt_z = ground_truth_df['translation_z'].values

        gt_x = ground_truth_df['translation_x'].values
        gt_y = ground_truth_df['translation_y'].values

        ground_truth_data = np.vstack((gt_x, gt_y, gt_z)).T.astype(np.float32)
        print("Ground truth data loaded successfully.")
        return ground_truth_data, gt_x, gt_y, gt_z
    except Exception as e:
        print(f"Error loading ground truth CSV from {gt_file_path}: {e}")
        return None, None, None, None