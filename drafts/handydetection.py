import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(21):
                x = int(hand_landmarks.landmark[i].x * frame.shape[1])
                y = int(hand_landmarks.landmark[i].y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
