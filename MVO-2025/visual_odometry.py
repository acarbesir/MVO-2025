import cv2
import numpy as np

def estimate_trajectory(video_path, camera_params_data, scale_factor=1.0,
                              display_live_trajectory_map=True, display_feature_matches=True):
    """
    Estimates the 3D (x, y, z) trajectory of an aerial vehicle from a bird's-eye view video.

    Args:
        video_path (str): Path to the input video file.
        camera_params_data (dict): A dictionary containing camera calibration parameters.
                                   Expected keys: 'IntrinsicMatrix', 'ImageSize', 'RadialDistortion', 'TangentialDistortion'.
        scale_factor (float): Multiplier for the estimated displacement, useful for tuning.
        display_live_trajectory_map (bool): If True, displays an accumulating 2D trajectory map using cv2.imshow.
        display_feature_matches (bool): If True, displays the feature matching sequence between frames.

    Returns:
        numpy.ndarray: An N x 3 array where N is the number of frames,
                       representing the estimated (x, y, z) trajectory.
    """

    # Extract camera intrinsic parameters
    intrinsic_matrix = camera_params_data['IntrinsicMatrix']
    image_size = tuple(camera_params_data['ImageSize']) # [height, width]
    # Combine radial and tangential distortion coefficients
    dist_coeffs = np.concatenate((camera_params_data['RadialDistortion'], camera_params_data['TangentialDistortion']))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=2000)

    # FLANN parameters for feature matching
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    trajectory = []

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None

    # Undistort the first frame
    prev_frame_undistorted = cv2.undistort(prev_frame, intrinsic_matrix, dist_coeffs)
    prev_gray = cv2.cvtColor(prev_frame_undistorted, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    # Initialize world coordinates (start from 0,0,0)
    current_x_world = 0.0
    current_y_world = 0.0
    # Set Z to a constant 10 meters
    current_z_world = 10.0
    trajectory.append([current_x_world, current_y_world, current_z_world])

    # --- Setup for Live Trajectory Map ---
    if display_live_trajectory_map:
        # Map size for display (width, height)
        traj_map_size = (600, 600)
        # Create a black image for drawing the trajectory
        traj = np.zeros((traj_map_size[1], traj_map_size[0], 3), dtype=np.uint8)
        # Offsets to center the trajectory display
        map_center_x_offset = 290
        map_center_y_offset = 290 # Adjust center for better visualization
    # --- End Setup ---

    frame_count = 0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Processing {int(total_frames)} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processing frame {frame_count}/{int(total_frames)}")

        # Undistort the current frame
        frame_undistorted = cv2.undistort(frame, intrinsic_matrix, dist_coeffs)
        gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)

        # Detect and compute ORB features for the current frame
        kp, des = orb.detectAndCompute(gray, None)

        # Handle cases where feature detection fails
        if prev_des is None or des is None or len(prev_kp) == 0 or len(kp) == 0:
            print(f"Warning: No features detected in frame {frame_count - 1} or {frame_count}. Skipping motion estimation.")
            # Append last known position if motion cannot be estimated
            trajectory.append([current_x_world, current_y_world, current_z_world])
            # Update display regardless
            if display_live_trajectory_map:
                _update_live_map(traj, current_x_world, current_y_world, current_z_world,
                                 frame_count, int(total_frames),
                                 map_center_x_offset, map_center_y_offset, traj_map_size, True)
            if display_feature_matches:
                _display_no_matches_message(frame_undistorted, "No features for matching")
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            # Update previous frame's data
            prev_gray = gray
            prev_kp = kp
            prev_des = des
            prev_frame_undistorted = frame_undistorted
            continue

        # Match features between current and previous frames
        matches = flann.knnMatch(prev_des, des, k=2)

        # Apply ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Handle cases with insufficient good matches
        if len(good_matches) < 8: # Minimum 8 points for Homography
            print(f"Warning: Not enough good matches found in frame {frame_count} ({len(good_matches)}). Skipping motion estimation.")
            trajectory.append([current_x_world, current_y_world, current_z_world])
            if display_live_trajectory_map:
                _update_live_map(traj, current_x_world, current_y_world, current_z_world,
                                 frame_count, int(total_frames),
                                 map_center_x_offset, map_center_y_offset, traj_map_size, True)
            if display_feature_matches:
                _display_feature_matches(prev_frame_undistorted, prev_kp, frame_undistorted, kp, matches, good_matches, mask_all=True)
                if len(good_matches) == 0: # If no good matches, display specific message
                     _display_no_matches_message(frame_undistorted, "Not enough good matches")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            prev_gray = gray
            prev_kp = kp
            prev_des = des
            prev_frame_undistorted = frame_undistorted
            continue

        # Extract points for homography estimation
        src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate Homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Handle failed homography estimation
        if H is None:
            print(f"Warning: Homography estimation failed for frame {frame_count}. Skipping.")
            trajectory.append([current_x_world, current_y_world, current_z_world])
            if display_live_trajectory_map:
                _update_live_map(traj, current_x_world, current_y_world, current_z_world,
                                 frame_count, int(total_frames),
                                 map_center_x_offset, map_center_y_offset, traj_map_size, True)
            if display_feature_matches:
                _display_no_matches_message(frame_undistorted, "Homography Failed")
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            prev_gray = gray
            prev_kp = kp
            prev_des = des
            prev_frame_undistorted = frame_undistorted
            continue

        # Calculate pixel shift of the image center
        center_x_prev = image_size[1] / 2
        center_y_prev = image_size[0] / 2
        center_pt_prev = np.array([[[center_x_prev, center_y_prev]]], dtype=np.float32)
        center_pt_current = cv2.perspectiveTransform(center_pt_prev, H)

        pixel_shift_x = center_pt_current[0, 0, 0] - center_x_prev
        pixel_shift_y = center_pt_current[0, 0, 1] - center_y_prev

        # Get focal lengths
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]

        # Convert pixel shift to real-world displacement (meters)
        # We use current_z_world (which is constant at 10.0m) for scaling X/Y displacement
        delta_x_world = pixel_shift_x * current_z_world / fx
        delta_y_world = pixel_shift_y * current_z_world / fy

        # Z-motion is now constant and initialized to 10.0, no change here
        # delta_z_world = 0.005 * scale_factor # This line is removed

        # Accumulate trajectory
        current_x_world += delta_x_world * scale_factor
        current_y_world += delta_y_world * scale_factor
        # current_z_world remains constant at 10.0, no accumulation here
        # current_z_world += delta_z_world # This line is removed

        trajectory.append([current_x_world, current_y_world, current_z_world])

        # --- Live Trajectory Map Display ---
        if display_live_trajectory_map:
            _update_live_map(traj, current_x_world, current_y_world, current_z_world,
                             frame_count, int(total_frames),
                             map_center_x_offset, map_center_y_offset, traj_map_size)

        # --- Feature Tracking Visualization ---
        if display_feature_matches:
            _display_feature_matches(prev_frame_undistorted, prev_kp, frame_undistorted, kp, matches, good_matches, mask)
        # --- End Feature Tracking Visualization ---

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Update previous frame data for the next iteration
        prev_gray = gray
        prev_kp = kp
        prev_des = des
        prev_frame_undistorted = frame_undistorted # Keep the undistorted frame for drawing matches

    cap.release()
    if display_live_trajectory_map or display_feature_matches:
        cv2.destroyAllWindows()
        if display_live_trajectory_map:
            cv2.imwrite('final_trajectory_map.png', traj)
            print("Final live trajectory map saved as 'final_trajectory_map.png'")

    print("Video processing complete.")
    return np.array(trajectory)

def _update_live_map(traj_img, current_x_world, current_y_world, current_z_world,
                     frame_count, total_frames,
                     map_center_x_offset, map_center_y_offset, traj_map_size, no_motion_warning=False):
    """Helper function to update and display the live trajectory map."""
    # Display estimated trajectory (note the sign reversal for visualization)
    # The visualization assumes a 2D bird's-eye view, so Z is not directly plotted here.
    display_x_est = -current_x_world
    display_y_est = -current_y_world

    draw_x_est = int(display_x_est) + map_center_x_offset
    draw_y_est = int(display_y_est) + map_center_y_offset

    # Clip coordinates to ensure they are within map boundaries
    draw_x_est = np.clip(draw_x_est, 0, traj_map_size[0] - 1)
    draw_y_est = np.clip(draw_y_est, 0, traj_map_size[1] - 1)

    color_est = (frame_count * 255 // total_frames, 255 - frame_count * 255 // total_frames, 0) # Greenish fade
    cv2.circle(traj_img, (draw_x_est, draw_y_est), 1, color_est, 1)

    # Display text overlays (current position, warnings)
    cv2.rectangle(traj_img, (10, 20), (590, 60), (0,0,0), -1) # Black background for text
    text_est = "VO: X=%5.2fm Y=%5.2fm Z=%5.2fm" % (display_x_est, display_y_est, current_z_world)
    cv2.putText(traj_img, text_est, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA) # Cyan for estimated

    if no_motion_warning:
        cv2.putText(traj_img, "Motion Est. Skipped!", (300, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Live Trajectory Map', traj_img)


def _display_feature_matches(prev_frame_undistorted, prev_kp, frame_undistorted, kp, matches, good_matches, mask=None, mask_all=False):
    """Helper function to display feature matches."""
    matches_to_draw = []

    if mask_all:
        # If mask_all is True, we want to draw all good_matches regardless of RANSAC result
        for m in good_matches:
            matches_to_draw.append([m])
    else:
        # Normal operation: Use the RANSAC mask to filter good_matches
        # The 'mask' array corresponds to the 'good_matches' list that was passed to findHomography
        # So, we iterate through good_matches and use the corresponding mask index.
        if mask is not None:
            for i, m in enumerate(good_matches):
                if mask[i] == 1: # Only draw if it's an inlier according to RANSAC
                    matches_to_draw.append([m])
        else: # If no mask is provided (e.g., in a scenario where homography wasn't computed)
              # or if you just want to draw all good_matches after ratio test without RANSAC filtering
            for m in good_matches:
                matches_to_draw.append([m])


    if len(matches_to_draw) > 0:
        feature_matches_img = cv2.drawMatchesKnn(prev_frame_undistorted, prev_kp,
                                                 frame_undistorted, kp,
                                                 matches_to_draw, None,
                                                 matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
                                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        feature_matches_img = np.zeros_like(frame_undistorted)
        cv2.putText(feature_matches_img, "No visible matches", (50, feature_matches_img.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Feature Matches', feature_matches_img)

def _display_no_matches_message(frame, message):
    """Helper function to display a message when no matches are found or homography fails."""
    no_matches_img = np.zeros_like(frame)
    cv2.putText(no_matches_img, message, (50, no_matches_img.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Feature Matches', no_matches_img)

