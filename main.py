import cv2
import csv
import numpy as np
import math

video_path = 'aruco_markers_data.csv'
# Initialize video capture
cap = cv2.VideoCapture('challengeB.mp4')

# Load ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
aruco_params = cv2.aruco.DetectorParameters()
# Output CSV file
csv_file = open(video_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame ID', 'Aruco ID', 'Aruco 2D', 'Aruco 3D'])

frame_id = 0

def calculate_3d_info(corners, frame_width, frame_height, marker_size=0.1):
    # Assuming the camera parameters for calculating the distance and yaw
    # These values are just for the sake of example, you need to calibrate your camera to get real values
    focal_length = 700  # Focal length in pixels
    real_marker_size = marker_size  # Real size of ArUco marker in meters

    # Calculate the distance to the camera
    marker_width = np.linalg.norm(corners[0] - corners[1])
    distance = (real_marker_size * focal_length) / marker_width

    # Calculate yaw angle with respect to the camera
    center_point = np.mean(corners, axis=0)
    yaw = math.degrees(math.atan2(center_point[0] - frame_width / 2, focal_length))

    return distance, yaw

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    frame_height, frame_width = frame.shape[:2]

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    if ids is not None:
        ids = ids.flatten()
        for i, corner in zip(ids, corners):
            corner = corner.reshape((4, 2))
            aruco_id = i % 1024  # Generate an ArUco ID for demonstration purposes

            # Calculate 3D information
            distance, yaw = calculate_3d_info(corner, frame_width, frame_height)

            # Write to CSV
            csv_writer.writerow([
                frame_id,
                aruco_id,
                corner.tolist(),
                [distance, yaw]
            ])

            # Draw the ArUco marker bounding box and ID on the frame
            for j in range(4):
                pt1 = (int(corner[j][0]), int(corner[j][1]))
                pt2 = (int(corner[(j + 1) % 4][0]), int(corner[(j + 1) % 4][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {aruco_id}", (int(corner[0][0]), int(corner[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()