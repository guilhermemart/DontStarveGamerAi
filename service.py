# Importar as bibliotecas necessárias
import torch
import torchvision
import cv2
import numpy as np
from PIL import ImageGrab
import time
from random import randrange as rand
from pathlib import Path

# Definir as classes do modelo "COCO"
classes = ['background', 'person', 'bicycle', 'car', 'motorcycle',
           'airplane', 'bus', 'train', 'truck', 'boat',
           'traffic_light', 'fyre_hidrant', 'street_sign', 'stop_sign', 'parking_meter',
           'bench', 'bird', 'cat', 'dog', 'horse', 
           'sheep', 'cow', 'elephant', 'bear', 'zebra', 
           'giraffe', 'hat', 'backpack', 'umbrella', 'shoe',
           'eye_glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
           'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle',
           'plate', 'wine_glass', 'cup', 'fork', 'knife',
           'spoon', 'bowl', 'banana', 'apple', 'sandwich',
           'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 
           'donut', 'cake', 'chair', 'couch', 'potted_plant', 
           'bed', 'mirror', 'dining_table', 'window', 'desk', 
           'toilet', 'door', 'tv', 'laptop', 'mouse',
           'remote', 'keyboard', 'cell_phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'blender', 'book',
           'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier',
           'toothbrush', 'hair_brush']

# Definir uma função para transformar a imagem em um tensor
def transform(image : cv2.typing.MatLike):
    # Redimensionar a imagem para 300x300 pixels
    image = cv2.resize(image, (600, 600))
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
def untransform_and_draw_boxes(image: torch.Tensor, boxes : tuple, labels : tuple, scores : tuple, threshold=0.5):
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
            # Se for person salvar como mini imagem
            if classes[label] == 'person':
                Path(__file__).parent.joinpath("data").joinpath("person").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f'{Path(__file__).parent.joinpath("data").joinpath("person").joinpath(f"pessoa{int(time.time())+rand(1,100,1)}.jpeg")}', image[y1:y2, x1:x2])
            # Desenhar a caixa na imagem
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Escrever o rótulo e o escore na imagem
            cv2.putText(image, f'{classes[label]}: {score:.2f}', (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

if __name__ == '__main__':

    # Carregar o modelo SSD pré-treinado
    if Path(__file__).parent.joinpath('model.pt').exists():
        model = torch.load('model.pt')
    else: 
        model = torchvision.models.detection.ssd300_vgg16(pretrained=True,weights='COCO_V1')

    model.training = True
    ima_= cv2.imread('C://Users//gmart//Projects//Ai//DontStarveGamerAi//data//person//pessoa.jpeg')
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    ima = [trans(ima_)]
    target = [{"boxes":torch.tensor(data=[0,0,600,600]), "labels":1}]
    model.forward(images = ima, targets = target)
    model.eval()
    
    # Capturar um print do monitor
    printscreen = ImageGrab.grab()
    # Converter o print de BGR para RGB e já transformar em um array numpy
    printscreen = cv2.cvtColor(np.array(printscreen), cv2.COLOR_BGR2RGB)

    # Transformar a imagem em um tensor
    image_tensor = transform(printscreen)

    # Fazer a predição com o modelo SSD
    with torch.no_grad():
        output = model(image_tensor)

    # Extrair as caixas delimitadoras, os rótulos e os escores da saída do modelo
    boxes = output[0]['boxes'].int().numpy()
    labels = output[0]['labels'].numpy()
    scores = output[0]['scores'].numpy()

    # Desenhar as caixas delimitadoras na imagem original
    image_with_boxes = untransform_and_draw_boxes(image_tensor[0], boxes, labels, scores)

    # salvar a imagem com as caixas delimitadoras
    cv2.imshow('Image with boxes', image_with_boxes)
    cv2.imwrite('Image_with_boxes.jpeg', image_with_boxes, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    torch.save(model, 'model.pt')
