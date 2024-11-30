import math

import cv2
import mss
import numpy as np


def capture_screenshot(region=None):
    """
    Captures a screenshot of the specified region of the screen.

    Args:
        region (dict): A dictionary specifying the region to capture. Should include:
                       {"top": y-coordinate, "left": x-coordinate, "width": w, "height": h}.
                       If None, captures the full screen.

    Returns:
        A screenshot as a numpy array in BGR format.
    """
    try:
        with mss.mss() as sct:
            # Capture the specified region or full screen if region is None
            screenshot = sct.grab(region if region else sct.monitors[1])

            # Convert to a numpy array and BGR format (for OpenCV compatibility)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            return frame
    except Exception as e:
        raise RuntimeError(f"Error during screenshot capture: {e}")


def detect_dominoes(frame):
    """
    Detects white dominoes in a given frame using color thresholding and edge detection.
    Focuses on the two faces of each domino and counts the dots in each face.

    Args:
        frame: The input image/frame (BGR format).

    Returns:
        A copy of the frame with detected dominoes outlined, division lines drawn,
        and dot counts displayed.
    """
    try:
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range for white color
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])

        # Create a mask for white color
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Copy the frame to draw rectangles
        output_frame = frame.copy()
        for contour in contours:
            # Get minimum area rectangle for the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            # Filter by size (adjust thresholds as needed)
            width, height = rect[1]
            width = math.ceil(width)
            height = math.ceil(height)
            if (width >= 10 and height >= 10) and (width <= 115 and height <= 115):
                # Draw the detected domino outline
                cv2.drawContours(output_frame, [box], 0, (0, 255, 0), 2)
                
                # Determine orientation (vertical or horizontal)
                vertical = width < height

                # Sort box points to identify corners
                box = sorted(box, key=lambda x: (x[1], x[0])) if vertical else sorted(
                    box, key=lambda x: (x[0], x[1]))
                
                # Calculate the midpoint to draw the division line
                midpoint1 = ((box[0][0] + box[1][0]) // 2, (box[0][1] + box[1][1]) // 2)
                midpoint2 = ((box[2][0] + box[3][0]) // 2, (box[2][1] + box[3][1]) // 2)
                cv2.line(output_frame, midpoint1, midpoint2, (255, 0, 0), 2)

                # Split the domino into two regions (faces)
                if vertical:  # Split vertically for horizontal domino vertical
                    face_width = (box[3][0] - box[0][0]) // 2
                    top_face = (box[0][0], box[0][1],
                                face_width, box[3][1] - box[0][1])
                    bottom_face = (box[0][0] + face_width, box[0]
                                   [1], face_width, box[3][1] - box[0][1])
                else:  # Split horizontally for vertical domino
                    face_height = (box[3][1] - box[0][1]) // 2
                    top_face = (box[0][0], box[0][1], box[3]
                                [0] - box[0][0], face_height)
                    bottom_face = (
                        box[0][0], box[0][1] + face_height, box[3][0] - box[0][0], face_height)

                # Draw red rectangles around the two faces
                for face in [top_face, bottom_face]:
                    x, y, w, h = face

                    # Ensure ROI coordinates are within bounds
                    x, y = max(0, x), max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)

                    cv2.rectangle(output_frame, (x + 3, y + 3),
                                  (x + w - 3, y + h - 3), (0, 0, 255), 2)

                    # Extract the ROI
                    roi = mask[y:y + h, x:x + w]

                    # Check if ROI is non-empty before processing
                    if roi.size == 0 or np.sum(roi) == 0:
                        continue

                    # Smooth the ROI
                    roi = cv2.GaussianBlur(roi, (5, 5), 0)

                    # Detect edges in the face using Canny edge detection algorithm
                    edges = cv2.Canny(
                        roi, 20, 75, apertureSize=5, L2gradient=True)

                    # Detect circles in the face
                    circles = cv2.HoughCircles(
                        edges,
                        cv2.HOUGH_GRADIENT,
                        dp=5.2,
                        minDist=10,
                        param1=80,
                        param2=30,
                        minRadius=1,
                        maxRadius=8,
                    )

                    dot_count = 0
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        dot_count = len(circles[0, :])

                        # Draw the detected circles
                        for circle in circles[0, :]:
                            center = (circle[0] + x, circle[1] + y)
                            radius = circle[2]
                            cv2.circle(output_frame, center,
                                       radius, (255, 0, 255), 2)

                    # Display the dot count on the face
                    cv2.putText(
                        output_frame,
                        str(dot_count),
                        (x + math.ceil(w / 3.2), y + math.ceil(h / 1.7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2,
                    )
        return output_frame

    except Exception as e:
        raise RuntimeError(f"Error during domino detection: {e}")
