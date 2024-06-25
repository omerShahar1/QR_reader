import cv2
import csv
import numpy as np
import math

# Initialize video capture
cap = cv2.VideoCapture('challengeB.mp4')

# QR code detector
detector = cv2.QRCodeDetector()

# Output CSV file
csv_file = open('qr_codes_data.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame ID', 'QR ID', 'QR 2D', 'QR 3D'])

frame_id = 0


def calculate_3d_info(points, frame_width, frame_height):
    # Assuming the camera parameters for calculating the distance and yaw
    # These values are just for the sake of example, you need to calibrate your camera to get real values
    focal_length = 700  # Focal length in pixels
    real_qr_size = 0.1  # Real size of QR code in meters

    # Calculate the distance to the camera
    qr_width = np.linalg.norm(points[0] - points[1])
    distance = (real_qr_size * focal_length) / qr_width

    # Calculate yaw angle with respect to the camera
    center_point = np.mean(points, axis=0)
    yaw = math.degrees(math.atan2(center_point[0] - frame_width / 2, focal_length))

    return distance, yaw


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    frame_height, frame_width = frame.shape[:2]

    # Detect QR codes
    retval, points = detector.detectMulti(frame)

    if retval:
        points = points.astype(int)
        for i, point in enumerate(points):
            # Decode QR code
            point_reshaped = point.reshape(4, 2)
            data, bbox, rectified_image = detector.detectAndDecode(frame)
            if data:
                qr_id = i % 1024  # Generate a QR ID for demonstration purposes

                # Calculate 3D information
                distance, yaw = calculate_3d_info(point_reshaped, frame_width, frame_height)

                # Write to CSV
                csv_writer.writerow([
                    frame_id,
                    qr_id,
                    point_reshaped.tolist(),
                    [distance, yaw]
                ])

                # Draw the QR code bounding box and ID on the frame
                for j in range(4):
                    cv2.line(frame, tuple(point_reshaped[j]), tuple(point_reshaped[(j + 1) % 4]), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {qr_id}", tuple(point_reshaped[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                            2)

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
