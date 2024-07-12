import cv2
import mediapipe as mp

# Inicializar o módulo de solução de MediaPipe para detecção de pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicializar a câmera
cap = cv2.VideoCapture(0)

# Tamanho médio dos ombros em metros (ajuste conforme necessário)
shoulder_width_meters = 0.4  # Por exemplo, 40 cm

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converter a imagem para colorida
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar a imagem e detectar pose
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Extrair landmarks
        landmarks = results.pose_landmarks.landmark

        # Calcular a distância entre ombros (por exemplo)
        shoulder_left = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
        shoulder_right = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)

        # Converter para coordenadas da imagem
        height, width, _ = frame.shape
        shoulder_left_px = (int(shoulder_left[0] * width), int(shoulder_left[1] * height))
        shoulder_right_px = (int(shoulder_right[0] * width), int(shoulder_right[1] * height))

        # Calcular a distância entre os ombros em pixels
        distance_px = abs(shoulder_right_px[0] - shoulder_left_px[0])

        # Calcular a distância do usuário à câmera
        # Usando uma relação simplificada entre tamanho do objeto na imagem e distância real
        # Ajuste conforme necessário com base na distância focal da câmera e na escala da imagem
        focal_length = 500  # Ajuste conforme necessário
        distance_to_camera = (shoulder_width_meters * focal_length) / distance_px

        # Desenhar um contorno ao redor do corpo
        for landmark in mp_pose.PoseLandmark:
            # Verificar se o landmark é visível na detecção
            if landmarks[landmark.value].visibility > 0.5:
                landmark_px = (int(landmarks[landmark.value].x * width), int(landmarks[landmark.value].y * height))
                cv2.circle(frame, landmark_px, 5, (0, 255, 0), -1)

        # Exibir a distância na tela
        cv2.putText(frame, f'Distance: {distance_to_camera:.2f} meters', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Exibir o frame
    cv2.imshow('Frame', frame)

    # Parar o loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar a janela
cap.release()
cv2.destroyAllWindows()
