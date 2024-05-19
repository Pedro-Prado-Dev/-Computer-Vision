import cv2
import numpy as np

# Caminhos para os arquivos de configuração e pesos
model_config = 'yolov3-tiny-face.cfg'
model_weights = 'yolov3-tiny-face.weights'
classes_file = 'face.names'

# Carregar as classes de objetos
with open(classes_file, 'r') as f:
    classes = f.read().strip().split('\n')

# Carregar a rede YOLO
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

# Definir o backend e a preferência do alvo
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Carregar uma imagem
image = cv2.imread('input.jpg')
height, width = image.shape[:2]

# Criar um blob a partir da imagem
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Obter os nomes das camadas de saída
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Executar a passagem direta (forward pass)
outputs = net.forward(output_layers)

# Analisar as detecções
boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Limite de confiança
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Coordenadas do retângulo de contorno
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Aplicar Non-Maximum Suppression (NMS)
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Desenhar as detecções na imagem
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        # Desenhar o retângulo do objeto detectado
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostrar a imagem resultante
cv2.imshow('YOLO Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
