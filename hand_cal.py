import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Mediapipe hand detector setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Calculator expression
expression = ""
last_click_time = 0

# Button class
class Button:
    def __init__(self, pos, text):
        self.pos = pos
        self.text = text
        self.size = 80

    def draw(self, img):
        x, y = self.pos
        cv2.rectangle(img, self.pos, (x + self.size, y + self.size), (255, 255, 255), cv2.FILLED)
        cv2.rectangle(img, self.pos, (x + self.size, y + self.size), (0, 0, 0), 2)
        cv2.putText(img, self.text, (x + 20, y + 55), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    def is_clicked(self, x, y):
        bx, by = self.pos
        return bx < x < bx + self.size and by < y < by + self.size


# Button layout
buttons = []
button_texts = [
    ["7", "8", "9", "+"],
    ["4", "5", "6", "-"],
    ["1", "2", "3", "*"],
    ["C", "0", "=", "/"]
]

for i in range(4):
    for j in range(4):
        buttons.append(Button((100 + j * 90, 100 + i * 90), button_texts[i][j]))

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Draw calculator buttons
    for button in buttons:
        button.draw(img)

    # Draw expression
    cv2.rectangle(img, (100, 30), (460, 80), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, expression, (110, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                lmList.append((int(lm.x * w), int(lm.y * h)))

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            if lmList:
                # Thumb tip and index tip
                x1, y1 = lmList[4]  # Thumb tip
                x2, y2 = lmList[8]  # Index tip

                # Show fingertips
                cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Check if fingers "pinch" (close together)
                distance = math.hypot(x2 - x1, y2 - y1)
                if distance < 40:  # Pinch gesture
                    current_time = time.time()
                    if current_time - last_click_time > 0.8:  # debounce
                        for button in buttons:
                            if button.is_clicked(x2, y2):
                                if button.text == "=":
                                    try:
                                        expression = str(eval(expression))
                                    except:
                                        expression = "Error"
                                elif button.text == "C":
                                    expression = ""
                                else:
                                    expression += button.text
                                last_click_time = current_time

    cv2.imshow("Finger Calculator", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
