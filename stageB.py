import cv2
import cv2.aruco as aruco
import numpy as np
import os


def find_aruco_markers(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    aruco_params = aruco.DetectorParameters()

    detected_corners, detected_ids, _ = aruco.detectMarkers(gray_img, aruco_dictionary, parameters=aruco_params)

    if detected_ids is not None:
        return detected_ids, detected_corners
    else:
        return [], []


def compute_error(reference, current):
    return np.linalg.norm(reference - current) / np.linalg.norm(reference) * 100


def calculate_marker_angle(marker_corners):
    vector_between_corners = marker_corners[1] - marker_corners[0]
    angle_rad = np.arctan2(vector_between_corners[1], vector_between_corners[0])
    return np.degrees(angle_rad)


def generate_movement_commands(ref_center, cur_center, ref_size, cur_size, ref_angle, cur_angle, tolerance=7):
    pos_error_x = ref_center[0] - cur_center[0]
    pos_error_y = ref_center[1] - cur_center[1]
    size_diff = (cur_size - ref_size) / ref_size * 100
    angle_diff = cur_angle - ref_angle

    movement_commands = []
    if abs(pos_error_x) > tolerance:
        if pos_error_x > 0:
            movement_commands.append('Move Right')
        else:
            movement_commands.append('Move Left')
    if abs(pos_error_y) > tolerance:
        if pos_error_y > 0:
            movement_commands.append('Move Down')
        else:
            movement_commands.append('Move Up')
    if abs(size_diff) > tolerance:
        if size_diff > 0:
            movement_commands.append('Move Backward')
        else:
            movement_commands.append('Move Forward')
    if abs(angle_diff) > tolerance:
        if angle_diff > 0:
            movement_commands.append('Turn Left')
        else:
            movement_commands.append('Turn Right')

    if not movement_commands:
        return "Matching", True

    return ", ".join(movement_commands), False


# Load the reference frame and detect the ArUco marker
ref_frame_path = 'C:\\New folder\\frame.png'
print(f"Attempting to load image from: {ref_frame_path}")

if not os.path.isfile(ref_frame_path):
    print(f"File not found: {ref_frame_path}")
    exit(1)

ref_frame = cv2.imread(ref_frame_path)

if ref_frame is None:
    print(f"Error: Unable to open image file {ref_frame_path}")
    exit(1)

ref_ids, ref_corners = find_aruco_markers(ref_frame)

if len(ref_ids) == 0:
    print("No ArUco marker found in the reference image.")
    exit(1)

# Assuming there's only one ArUco marker in the reference image
ref_marker_id = ref_ids[0][0]
ref_marker_corners = ref_corners[0][0]
ref_marker_center = np.mean(ref_marker_corners, axis=0)
ref_marker_size = np.linalg.norm(ref_marker_corners[0] - ref_marker_corners[2])
ref_marker_angle = calculate_marker_angle(ref_marker_corners)

# Get reference frame dimensions
ref_img_height, ref_img_width = ref_frame.shape[:2]

# Open video stream with camera set to match reference frame dimensions
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit(1)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, ref_img_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, ref_img_height)

while video_capture.isOpened():
    ret, live_frame = video_capture.read()
    if not ret:
        break

    current_ids, current_corners = find_aruco_markers(live_frame)

    if len(current_ids) > 0 and ref_marker_id in current_ids:
        idx = np.where(current_ids == ref_marker_id)[0][0]
        curr_marker_id = current_ids[idx][0]
        curr_marker_corners = current_corners[idx][0]
        curr_marker_center = np.mean(curr_marker_corners, axis=0)
        curr_marker_size = np.linalg.norm(curr_marker_corners[0] - curr_marker_corners[2])
        curr_marker_angle = calculate_marker_angle(curr_marker_corners)

        command, is_matching = generate_movement_commands(ref_marker_center, curr_marker_center, ref_marker_size,
                                                          curr_marker_size, ref_marker_angle, curr_marker_angle,
                                                          tolerance=20)
        cv2.putText(live_frame, f"ID: {curr_marker_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(live_frame, command, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if is_matching else (0, 0, 255), 2)

        if is_matching:
            cv2.putText(live_frame, "Matching", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(live_frame, "ArUco ID not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Live Feed', live_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

