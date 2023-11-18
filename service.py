# Importar as bibliotecas necessárias
import torch
import torchvision
import cv2
import numpy as np
from PIL import ImageGrab
import time
from random import randrange as rand
from pathlib import Path
import os

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
def untransform_and_draw_boxes(tensor_image: torch.Tensor, boxes : tuple, labels : tuple, scores : tuple, threshold=0.5):
    image = tensor_image.clone()
    # Converter a imagem em um array numpy
    image = image.numpy()
    # Transpor a imagem para o formato HxWxC
    image = image.transpose(1, 2, 0)
    # Desnormalizar a imagem
    image = image * 255.0
    # Iterar sobre as caixas, os rótulos e os escores
    are_there_persons = []
    for box, label, score in zip(boxes, labels, scores):
        # Verificar se o escore é maior que o limiar
        if score > threshold:
            # Extrair as coordenadas da caixa
            x1, y1, x2, y2 = box
            # Se for person separar a miniimagem
            if classes[label] == 'person':
                Path(__file__).parent.joinpath("data").joinpath("person").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f'{Path(__file__).parent.joinpath("data").joinpath("person").joinpath(f"pessoa{int(time.time())+rand(1,100,1)}.jpeg")}', image[y1:y2, x1:x2])
                are_there_persons.append(box)
            # Desenhar a caixa na imagem
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Escrever o rótulo e o escore na imagem
            cv2.putText(image, f'{classes[label]}: {score:.2f}', (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image, are_there_persons

def train(model : torchvision.models.detection.ssd.SSD, file_dir ='C://Users//gmart//Projects//Ai//DontStarveGamerAi//data//person', epochs = 1):
    model.training = True

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    for _ in range(epochs):
        files = os.listdir(file_dir)
        for file in files:
            ima_= cv2.imread(file_dir + '//' + file)
            trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            ima = [trans(ima_)]
            target = [{"boxes":torch.tensor(data=[[0,0,600,600]]), "labels":torch.tensor(data=[1])}]

            optimizer.zero_grad()
            output = model.forward(images = ima, targets = target) 
            loss = output.get('bbox_regression', None)
            
            if loss is not None:
                loss.backward()
                optimizer.step()
            
            
    model.eval()
    return model