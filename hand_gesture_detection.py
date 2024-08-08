import numpy as np
import mediapipe as mp
import cv2
import pyautogui
import random
from pynput.mouse import Button, Controller

def get_angle(a, b, c):
    # Calculate the angle between three points
    # a, b, c are points represented as (x, y) tuples
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    # Convert radians to degrees and take absolute value
    angle = np.abs(np.degrees(radians))
    return angle

def get_distance(landmark_list):
    # Calculate distance between two landmarks
    if len(landmark_list) < 2:
        return
    # Extract coordinates of two points
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    # Calculate Euclidean distance
    L = np.hypot(x2-x1, y2-y1)
    # Interpolate distance to a 0-1000 range
    return np.interp(L, [0, 1], [0, 1000])

# Initialize mouse controller
mouse = Controller()

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,  # Set to False for video streams
    model_complexity=1,       # 0 is fastest, 1 is more accurate
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1           # Detect only one hand
)

def find_finger_tip(processed):
    # Extract the position of the index finger tip
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        # Return the landmark for the index finger tip
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None

def move_mouse(index_finger_tip):
    # Move the mouse cursor based on index finger tip position
    if index_finger_tip is not None:
        # Convert normalized coordinates to screen coordinates
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(x, y)

def is_left_click(landmarks_list, thumb_index_dist):
    # Detect left click gesture
    return (get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and 
            get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) > 90 and
            thumb_index_dist > 50)

def is_right_click(landmarks_list, thumb_index_dist):
    # Detect right click gesture
    return (get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 90 and 
            get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
            thumb_index_dist > 50)

def is_double_click(landmarks_list, thumb_index_dist):
    # Detect double click gesture
    return (get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and 
            get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
            thumb_index_dist > 50)

def is_screenshot(landmarks_list, thumb_index_dist):
    # Detect screenshot gesture
    return (get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and 
            get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
            thumb_index_dist < 50)

def detect_gestures(frame, landmarks_list, processed):
    if len(landmarks_list) >= 21:  # Ensure all hand landmarks are detected
        index_finger_tip = find_finger_tip(processed)
        # Calculate distance between thumb tip and index finger base
        thumb_index_dist = get_distance([landmarks_list[4], landmarks_list[5]])

        # Detect mouse movement
        if thumb_index_dist < 50 and get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 90:
            move_mouse(index_finger_tip)

        # Detect left click
        elif is_left_click(landmarks_list, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Detect right click
        elif is_right_click(landmarks_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Detect double click
        elif is_double_click(landmarks_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Detect screenshot
        elif is_screenshot(landmarks_list, thumb_index_dist):
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'py_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Hand not detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            # Convert the BGR image to RGB
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and find hand landmarks
            processed = hands.process(frameRGB)

            landmarks_list = []

            # If hand landmarks are detected
            if processed.multi_hand_landmarks:
                # Get the first detected hand
                hand_landmarks = processed.multi_hand_landmarks[0]
                # Draw the hand landmarks on the frame
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

                # Store the landmark coordinates
                for lm in hand_landmarks.landmark:
                    landmarks_list.append((lm.x, lm.y))

            # Detect and process hand gestures
            detect_gestures(frame, landmarks_list, processed)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    finally:
        # Release the capture and destroy all windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()