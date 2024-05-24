import cv2
import torch
import audio


#Carregar o modelo YOLOv5 pré-treinado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

#Abrir a câmera
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
    boxes = results.xyxy[0].numpy()  # caixes xyxy
    labels = results.names

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = labels[int(cls)]

        # Desenhar a caixa delimitadora
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Se o objeto detectado for uma pessoa, calcular a distância aproximada
        if label == 'person':
            distance = (2 * 3.14 * 180) / (x2 - x1 + y2 - y1) * 1000 + 3  # fórmula fictícia para calcular distância
            distance_m = distance / 100  # converter para metros
            cv2.putText(frame, f'Distance: {distance_m:.2f} m', (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (36, 255, 12), 2)
            audio.Speaker.speak(f"Pessoa a {distance_m:.2f}")

    # Mostrar o frame processado
    cv2.imshow('YOLOv5 Object Detection', frame)


cap.release()
cv2.destroyAllWindows()