import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from enum import Enum
import logging
import sys

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
SMOOTHING_FACTOR = 0.3
SCROLL_SPEED = 20
CLICK_DISTANCE_THRESHOLD = 0.05
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
DEBOUNCE_TIME = 0.2

class Gesture(Enum):
    NONE = 0
    SCROLL_UP = 1
    SCROLL_DOWN = 2
    MOVE_CURSOR = 3
    CLICK = 4
    RIGHT_CLICK = 5

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_click_time = 0
        self.prev_x, self.prev_y = None, None
        pyautogui.FAILSAFE = True
        self.current_action = ""
        logger.info("GestureController initialized")

    def get_distance(self, point1, point2):
        return np.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)

    def detect_gesture(self, landmarks):
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        index_up = index_tip.y < landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle_up = middle_tip.y < landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_up = ring_tip.y < landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_up = pinky_tip.y < landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y
        thumb_down = thumb_tip.y > landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y

        # Click Detection
        if self.get_distance(thumb_tip, index_tip) < CLICK_DISTANCE_THRESHOLD:
            if time.time() - self.last_click_time > DEBOUNCE_TIME:
                self.last_click_time = time.time()
                self.current_action = "Click"
                return Gesture.CLICK

        # Right Click Detection
        if self.get_distance(thumb_tip, middle_tip) < CLICK_DISTANCE_THRESHOLD:
            if time.time() - self.last_click_time > DEBOUNCE_TIME:
                self.last_click_time = time.time()
                self.current_action = "Right Click"
                return Gesture.RIGHT_CLICK

        # Scroll Gesture
        if index_up and middle_up and ring_up and not pinky_up and thumb_down:
            if index_tip.y < 0.5:
                self.current_action = "Scroll Up"
                return Gesture.SCROLL_UP
            else:
                self.current_action = "Scroll Down"
                return Gesture.SCROLL_DOWN

        # Cursor Move
        if (index_up and not middle_up and not ring_up) or (index_up and middle_up and not ring_up):
            self.current_action = "Move Cursor"
            return Gesture.MOVE_CURSOR

        self.current_action = ""
        return Gesture.NONE
    def execute_gesture(self, gesture, landmarks):
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        if gesture == Gesture.MOVE_CURSOR:
            # Map coordinates (with amplified motion zone)
            screen_x = np.interp(index_tip.x, [0.2, 0.8], [0, SCREEN_WIDTH])
            screen_y = np.interp(index_tip.y, [0.2, 0.8], [0, SCREEN_HEIGHT])

            # Faster smoothing
            fast_smooth_factor = 0.1  # Lower = faster response

            if self.prev_x is not None and self.prev_y is not None:
                screen_x = self.prev_x + (screen_x - self.prev_x) * fast_smooth_factor
                screen_y = self.prev_y + (screen_y - self.prev_y) * fast_smooth_factor

            pyautogui.moveTo(screen_x, screen_y)
            self.prev_x, self.prev_y = screen_x, screen_y

        elif gesture == Gesture.SCROLL_UP:
            pyautogui.scroll(SCROLL_SPEED * 3)  # 3x faster scroll

        elif gesture == Gesture.SCROLL_DOWN:
            pyautogui.scroll(-SCROLL_SPEED * 3)  # 3x faster scroll

        elif gesture == Gesture.CLICK:
            pyautogui.click()

        elif gesture == Gesture.RIGHT_CLICK:
            pyautogui.rightClick()
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        # Finger states
        index_up = index_tip.y < landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle_up = middle_tip.y < landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_up = ring_tip.y < landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_up = pinky_tip.y < landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y
        thumb_down = thumb_tip.y > landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y

        # Click Detection
        if self.get_distance(thumb_tip, index_tip) < CLICK_DISTANCE_THRESHOLD:
            if time.time() - self.last_click_time > DEBOUNCE_TIME:
                self.last_click_time = time.time()
                self.current_action = "Click"
                return Gesture.CLICK

        # Right Click
        if self.get_distance(thumb_tip, middle_tip) < CLICK_DISTANCE_THRESHOLD:
            if time.time() - self.last_click_time > DEBOUNCE_TIME:
                self.last_click_time = time.time()
                self.current_action = "Right Click"
                return Gesture.RIGHT_CLICK

        # 🧠 3-FINGER SCROLL
        if index_up and middle_up and ring_up and not pinky_up and thumb_down:
            if index_tip.y < 0.5:
                self.current_action = "Scroll Up (3-Fingers)"
                return Gesture.SCROLL_UP
            else:
                self.current_action = "Scroll Down (3-Fingers)"
                return Gesture.SCROLL_DOWN

        # Cursor Move with index only OR index + middle
        if index_up and not middle_up and not ring_up:
            self.current_action = "Move Cursor (Index Only)"
            return Gesture.MOVE_CURSOR
        if index_up and middle_up and not ring_up:
            self.current_action = "Move Cursor (2 Fingers)"
            return Gesture.MOVE_CURSOR

        self.current_action = ""
        return Gesture.NONE

    def draw_pointers(self, frame, landmarks):
        h, w, _ = frame.shape
        index = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        middle = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        # Convert normalized coordinates to pixel coordinates
        ix, iy = int(index.x * w), int(index.y * h)
        tx, ty = int(thumb.x * w), int(thumb.y * h)
        mx, my = int(middle.x * w), int(middle.y * h)

        # Draw Index Finger Dot
        cv2.circle(frame, (ix, iy), 10, (0, 255, 255), -1)

        # Draw Thumb Dot
        if self.get_distance(index, thumb) < CLICK_DISTANCE_THRESHOLD:
            cv2.circle(frame, (tx, ty), 10, (0, 255, 0), -1)  # Green when touching
        else:
            cv2.circle(frame, (tx, ty), 10, (0, 0, 255), -1)  # Red when not

        # Draw Middle Finger Dot
        if self.get_distance(thumb, middle) < CLICK_DISTANCE_THRESHOLD:
            cv2.circle(frame, (mx, my), 10, (0, 255, 0), -1)  # Green for right-click
        else:
            cv2.circle(frame, (mx, my), 10, (255, 0, 0), -1)

        # Action text
        if self.current_action:
            cv2.putText(frame, f"Action: {self.current_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (50, 255, 50), 2)

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                gesture = self.detect_gesture(hand_landmarks)
                self.execute_gesture(gesture, hand_landmarks)
                self.draw_pointers(frame, hand_landmarks)

        return frame

    def cleanup(self):
        self.hands.close()
        logger.info("Cleaned up MediaPipe")

def main():
    controller = GestureController()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Webcam not available")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame = controller.process_frame(frame)
            cv2.imshow("Finger Control with Feedback", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        controller.cleanup()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
