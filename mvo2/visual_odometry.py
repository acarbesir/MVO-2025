import cv2
import numpy as np
import os


def estimate_trajectory(
        frames_folder_path,
        camera_params_data,
        gt_z_data=None,  # New parameter to accept ground truth Z
        z_constant_fallback=0.0,  # Fallback Z for when GT is not available or exhausted
        scale_factor=1.0,
        display_live_trajectory_map=True,  # kept for backward-compatibility (ignored)
        display_feature_matches=True,
        ground_truth_data_for_display=None,  # Not directly used for calculation here, but kept for clarity
):
    """
    Estimates the 2D (x, y) trajectory of an aerial vehicle from a sequence of
    frames stored in a folder, incorporating dynamic Z-axis information
    from ground truth where available.

    **Note**: The *live map* option has been **removed**. The parameter
    `display_live_trajectory_map` is retained so that existing calls to this
    function do not break, but it is ignored internally.

    Args
    ----
    frames_folder_path (str): Path to the folder containing image frames.
    camera_params_data (dict): Camera calibration parameters with keys
        `'IntrinsicMatrix', 'ImageSize', 'RadialDistortion',
        `'TangentialDistortion`.
    gt_z_data (np.ndarray, optional): N-element array of ground truth Z-coordinates.
                                      Used to dynamically adjust scale when available.
                                      Defaults to None.
    z_constant_fallback (float): Constant height (z-coordinate) of the vehicle
                                 to use when ground truth Z is not available.
    scale_factor (float): Multiplier for the estimated displacement (applied after
                          Z-based scaling, for overall fine-tuning if needed).
    display_live_trajectory_map (bool): *Deprecated* – ignored.
    display_feature_matches (bool): If `True` shows feature matches with
        `cv2.imshow`.
    ground_truth_data_for_display (np.ndarray): Optional N×3 array of ground
        truth poses for overlay on the feature-match view (not on a live
        map – the map is no longer drawn).

    Returns
    -------
    np.ndarray
    An N × 3 array representing the estimated (x, y, z) trajectory.
    """

    # ------------------------------------------------------------------
    # 0. Sanity checks & pre-processing
    # ------------------------------------------------------------------
    if not os.path.isdir(frames_folder_path):
        print(f"Error: Frames folder '{frames_folder_path}' not found.")
        return None

    image_files = sorted(
        [
            f
            for f in os.listdir(frames_folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))
        ]
    )
    if not image_files:
        print(f"Error: No image files found in '{frames_folder_path}'.")
        return None

    intrinsic_matrix = camera_params_data["IntrinsicMatrix"]
    image_size = tuple(camera_params_data["ImageSize"])  # [height, width]
    dist_coeffs = np.concatenate(
        (
            camera_params_data["RadialDistortion"],
            camera_params_data["TangentialDistortion"],
        )
    )

    # ORB detector
    orb = cv2.ORB_create(nfeatures=2000)

    # FLANN matcher
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # ------------------------------------------------------------------
    # 1. Initialisation
    # ------------------------------------------------------------------
    trajectory = []

    first_path = os.path.join(frames_folder_path, image_files[0])
    first_frame = cv2.imread(first_path)
    if first_frame is None:
        print(f"Error: Could not read the first frame from {first_path}.")
        return None

    first_undist = cv2.undistort(first_frame, intrinsic_matrix, dist_coeffs)
    prev_gray = cv2.cvtColor(first_undist, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    current_x_world = 0.0
    current_y_world = 0.0

    # Initialize current_z_world based on GT or fallback
    current_z_world = z_constant_fallback
    if gt_z_data is not None and len(gt_z_data) > 0:
        current_z_world = gt_z_data[0]  # Use initial GT Z if available

    trajectory.append([current_x_world, current_y_world, current_z_world])

    frame_count = 0
    total_frames = len(image_files)
    print(f"Processing {total_frames} frames…")

    # ------------------------------------------------------------------
    # 2. Main loop – per-frame processing
    # ------------------------------------------------------------------
    for i in range(1, total_frames):
        path = os.path.join(frames_folder_path, image_files[i])
        frame = cv2.imread(path)
        if frame is None:
            print(f"Warning: Could not read frame {path}. Skipping.")
            # Append current known trajectory if frame missing
            trajectory.append([current_x_world, current_y_world, current_z_world])
            frame_count += 1
            continue

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")

        undist = cv2.undistort(frame, intrinsic_matrix, dist_coeffs)
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        # --------------------------------------------------------------
        # 2.1 Handle cases with no / bad keypoints
        # --------------------------------------------------------------
        if (
                prev_des is None
                or des is None
                or len(prev_kp) == 0
                or len(kp) == 0
        ):
            print(
                f"Warning: No features detected in frame {frame_count - 1} or {frame_count}. Skipping motion estimation."
            )
            # Append current known trajectory if features are insufficient
            trajectory.append([current_x_world, current_y_world, current_z_world])

            if display_feature_matches:
                _display_no_matches_message(undist, "No features for matching")
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            # Update "previous" references and continue
            prev_gray, prev_kp, prev_des = gray, kp, des
            # prev_frame = undist # Keep prev_frame consistent for _display_feature_matches
            continue

        # --------------------------------------------------------------
        # 2.2 Feature matching & ratio test
        # --------------------------------------------------------------
        matches = flann.knnMatch(prev_des, des, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) < 8:
            print(
                f"Warning: Not enough good matches in frame {frame_count} ({len(good_matches)}). Skipping motion estimation."
            )
            # Append current known trajectory if matches are insufficient
            trajectory.append([current_x_world, current_y_world, current_z_world])

            if display_feature_matches:
                _display_feature_matches(  # Display with the few matches if any
                    prev_frame=first_undist, prev_kp=prev_kp,
                    frame_curr=undist, kp_curr=kp,
                    matches=matches, good_matches=good_matches, mask_all=True
                )
                if len(good_matches) == 0:  # If literally no good matches
                    _display_no_matches_message(undist, "Not enough good matches")
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            prev_gray, prev_kp, prev_des = gray, kp, des
            # prev_frame = undist # Keep prev_frame consistent
            continue

        # --------------------------------------------------------------
        # 2.3 Homography estimation
        # --------------------------------------------------------------
        src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            print(f"Warning: Homography failed for frame {frame_count}. Skipping.")
            # Append current known trajectory if homography failed
            trajectory.append([current_x_world, current_y_world, current_z_world])

            if display_feature_matches:
                _display_no_matches_message(undist, "Homography Failed")
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            prev_gray, prev_kp, prev_des = gray, kp, des
            # prev_frame = undist # Keep prev_frame consistent
            continue

        # --------------------------------------------------------------
        # 2.4 Pixel-shift → world displacement
        # --------------------------------------------------------------
        cx_prev, cy_prev = image_size[1] / 2, image_size[0] / 2
        center_prev = np.array([[[cx_prev, cy_prev]]], dtype=np.float32)
        center_curr = cv2.perspectiveTransform(center_prev, H)

        dx_px = center_curr[0, 0, 0] - cx_prev
        dy_px = center_curr[0, 0, 1] - cy_prev

        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]

        use_poly_z_model = False
        z_poly_model = None

        if gt_z_data is not None and len(gt_z_data) >= 10:
            n_calib = min(450, len(gt_z_data))
            z_times = np.arange(n_calib)
            z_values = gt_z_data[:n_calib]

            # Fit a polynomial to the GT Z values
            poly_degree = 2  # You can change this to 3 if needed
            z_poly_model = np.poly1d(np.polyfit(z_times, z_values, poly_degree))
            use_poly_z_model = True

        if gt_z_data is not None and i < 450 and i < len(gt_z_data):
            z_value_for_current_frame = gt_z_data[i]
        elif use_poly_z_model:
            z_value_for_current_frame = z_poly_model(i)
        else:
            z_value_for_current_frame = current_z_world  # fallback (last known or constant)

        dx_m = dx_px / fx
        dy_m = dy_px / fy

        current_x_world += dx_m * scale_factor
        current_y_world += dy_m * scale_factor
        # The Z value for this frame's trajectory point will be the one used for calculation
        current_z_world = z_value_for_current_frame

        trajectory.append([current_x_world, current_y_world, current_z_world])

        # --------------------------------------------------------------
        # 2.5 Feature-match visualisation (optional)
        # --------------------------------------------------------------
        if display_feature_matches:
            _display_feature_matches(prev_frame=first_undist, prev_kp=prev_kp,
                                     frame_curr=undist, kp_curr=kp,
                                     matches=matches, good_matches=good_matches,
                                     mask=mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Update previous (important: prev_frame needs to be the undistorted version of the previous actual frame)
        prev_gray, prev_kp, prev_des, first_undist = gray, kp, des, undist

    # ------------------------------------------------------------------
    # 3. Cleanup
    # ------------------------------------------------------------------
    if display_feature_matches:
        cv2.destroyAllWindows()

    print("Frame processing complete.")
    return np.array(trajectory)


# ----------------------------------------------------------------------
# Helper visualisation functions (unchanged)
# ----------------------------------------------------------------------

def _display_feature_matches(
        prev_frame,
        prev_kp,
        frame_curr,
        kp_curr,
        matches,
        good_matches,
        mask=None,
        mask_all=False,
):
    """Display feature matches using `cv2.drawMatchesKnn`."""
    matches_to_draw = []

    if mask_all:
        matches_to_draw = [[m] for m in good_matches]
    else:
        if mask is not None:
            for i, m in enumerate(good_matches):
                if mask[i] == 1:
                    matches_to_draw.append([m])
        else:
            matches_to_draw = [[m] for m in good_matches]

    if matches_to_draw:
        vis = cv2.drawMatchesKnn(
            prev_frame,
            prev_kp,
            frame_curr,
            kp_curr,
            matches_to_draw,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    else:
        vis = np.zeros_like(frame_curr)
        cv2.putText(
            vis,
            "No visible matches",
            (50, vis.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Feature Matches", vis)


def _display_no_matches_message(frame, message):
    """Show a placeholder image with an error/warning message."""
    placeholder = np.zeros_like(frame)
    cv2.putText(
        placeholder,
        message,
        (50, placeholder.shape[0] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Feature Matches", placeholder)