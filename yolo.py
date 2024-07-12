import cv2
import torch
import pyttsx3
import time
import threading

# Carregar o modelo YOLOv5 pré-treinado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Abrir a câmera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

# Inicializar o pyttsx3 para conversão de texto para fala
engine = pyttsx3.init()

# Configurações de voz
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Seleciona uma voz feminina

last_spoken_time = time.time() - 2  # Inicializa com um tempo passado para falar imediatamente
lock = threading.Lock()

def speak_text(text):
    with lock:
        engine.say(text)
        engine.runAndWait()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame")
        break

    # Converter a imagem BGR para RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizar a detecção
    results = model(img)

    # Obter as caixas delimitadoras e os rótulos
    boxes = results.xyxy[0].cpu().numpy()  # caixas xyxy
    labels = results.names

    person_detected = False
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = labels[int(cls)]

        # Se o objeto detectado for uma pessoa, calcular a distância aproximada
        if label == 'person':
            # Fórmula fictícia para calcular a distância (baseada no tamanho da caixa delimitadora)
            distance = 650 / (x2 - x1 + y2 - y1)  # ajuste conforme necessário
            # Debug: Print distance to console
            print(f'Person detected at distance: {distance:.2f} m')
            distance_text = f'Distance: {distance:.2f} meters'
            person_detected = True

            # Verificar se a distância está em 100cm, 200cm ou 300cm
            if int(distance) in [1, 2, 3]:
                # Verificar se já passaram 2 segundos desde o último aviso sonoro
                current_time = time.time()
                if current_time - last_spoken_time >= 2:
                    # Converter a distância para texto e falar
                    threading.Thread(target=speak_text, args=(f'Distance to person is {distance:.2f} meters',)).start()
                    last_spoken_time = current_time  # Atualiza o tempo do último aviso sonoro

        # Desenhar a caixa delimitadora
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Desenhar apenas o texto da identificação
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Se nenhuma pessoa foi detectada, reiniciar o tempo do último aviso sonoro
    if not person_detected:
        last_spoken_time = time.time() - 2

    # Mostrar o frame processado
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Parar se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
