# Importar as bibliotecas necessárias
import torch
import torchvision
import cv2
import numpy as np

# Carregar o modelo SSD pré-treinado
model = torchvision.models.detection.ssd300_vgg16(pretrained=True,weights='COCO_V1')
model.eval()

# Definir as classes do modelo
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tvmonitor']

# Definir uma função para transformar a imagem em um tensor
def transform(image):
    # Redimensionar a imagem para 300x300 pixels
    image = cv2.resize(image, (300, 300))
    # Converter a imagem em um tensor
    image = torch.from_numpy(image).float()
    # Normalizar a imagem
    image = image / 255.0
    # Transpor a imagem para o formato CxHxW
    image = image.permute(2, 0, 1)
    # Adicionar uma dimensão de lote
    image = image.unsqueeze(0)
    return image

# Definir uma função para desenhar as caixas delimitadoras na imagem
def draw_boxes(image, boxes, labels, scores, threshold=0.5):
    # Converter a imagem em um array numpy
    image = image.numpy()
    # Transpor a imagem para o formato HxWxC
    image = image.transpose(1, 2, 0)
    # Desnormalizar a imagem
    image = image * 255.0
    # Iterar sobre as caixas, os rótulos e os escores
    for box, label, score in zip(boxes, labels, scores):
        # Verificar se o escore é maior que o limiar
        if score > threshold:
            # Extrair as coordenadas da caixa
            x1, y1, x2, y2 = box
            # Desenhar a caixa na imagem
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Escrever o rótulo e o escore na imagem
            cv2.putText(image, f'{classes[label]}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

# Carregar uma imagem de exemplo
image = cv2.imread('aviao.jpeg')

# Transformar a imagem em um tensor
image_tensor = transform(image)

# Fazer a predição com o modelo SSD
with torch.no_grad():
    output = model(image_tensor)

# Extrair as caixas delimitadoras, os rótulos e os escores da saída do modelo
boxes = output[0]['boxes'].int().numpy()
labels = output[0]['labels'].numpy()
scores = output[0]['scores'].numpy()

# Desenhar as caixas delimitadoras na imagem original
image_with_boxes = draw_boxes(image_tensor[0], boxes, labels, scores)

# Mostrar a imagem com as caixas delimitadoras
cv2.imshow('Image with boxes', image_with_boxes)
cv2.imwrite('aviao2.jpeg', image_with_boxes, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
cv2.waitKey(0)
cv2.destroyAllWindows()
