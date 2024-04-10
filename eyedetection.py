import cv2
import mediapipe as mp

# Inicializar o detector de rosto do MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Inicializar o detector de olhos do MediaPipe
mp_eye_detection = mp.solutions.face_mesh
eye_detection = mp_eye_detection.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Inicializar a webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter a imagem para RGB para o MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar rostos na imagem
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Converter para RGB para o MediaPipe
            rgb_face = frame[y:y+h, x:x+w]

            # Detectar olhos na imagem
            eyes_results = eye_detection.process(rgb_face)

            if eyes_results.multi_face_landmarks:
                for face_landmarks in eyes_results.multi_face_landmarks:
                    for eye in face_landmarks.landmark[160:168]:  # Índices dos pontos dos olhos
                        eye_x = int(eye.x * w) + x
                        eye_y = int(eye.y * h) + y
                        cv2.circle(frame, (eye_x, eye_y), 2, (0, 255, 255), -1)

    cv2.imshow('Eye Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
