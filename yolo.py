import cv2
import torch
import numpy as np

# Carregar o modelo YOLOv5 pré-treinado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Abrir a câmera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

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

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = labels[int(cls)]

        # Desenhar a caixa delimitadora
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Se o objeto detectado for uma pessoa, calcular a distância aproximada
        if label == 'person':
            # Fórmula fictícia para calcular a distância (baseada no tamanho da caixa delimitadora)
            distance = 500 / (x2 - x1 + y2 - y1)  # ajuste conforme necessário
            # Debug: Print distance to console
            print(f'Person detected at distance: {distance:.2f} m')
            distance_text = f'Distance: {distance:.2f} m'
            # Desenhar o texto da identificação e da distância
            cv2.putText(frame, label, (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(frame, distance_text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)
        else:
            # Desenhar apenas o texto da identificação
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Mostrar o frame processado
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Parar se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
