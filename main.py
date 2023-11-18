import torch
import torchvision
import cv2
import numpy as np
from PIL import ImageGrab
import time
from random import randrange as rand
from pathlib import Path
import service

if __name__ == '__main__':

    # Carregar o modelo SSD pré-treinado
    if Path(__file__).parent.joinpath('model.pt').exists():
        model = torch.load('model.pt')
    else: 
        model = torchvision.models.detection.ssd300_vgg16(pretrained=True,weights='COCO_V1')

    # Carregar o modelo SSD personalizado
    if Path(__file__).parent.joinpath('model_local.pt').exists():
        model_local = torch.load('model_local.pt')
    else: 
        model_local = torchvision.models.detection.ssd300_vgg16(pretrained=False,num_classes=2)

    # Treinar o model COCO para imagens locais
    model = service.train(model, 'C://Users//gmart//Projects//Ai//DontStarveGamerAi//data//person')

    # Treinar o model local para imagens heads
    model_local = service.train(model_local, 'C://Users//gmart//Projects//Ai//DontStarveGamerAi//data//head', 1)
    
    # Capturar um print do monitor
    printscreen = ImageGrab.grab()
    # Converter o print de BGR para RGB e já transformar em um array numpy
    printscreen = cv2.cvtColor(np.array(printscreen), cv2.COLOR_BGR2RGB)

    # Transformar a imagem em um tensor
    image_tensor = service.transform(printscreen)

    # Fazer a predição com o modelo SSD
    with torch.no_grad():
        output = model(image_tensor)

    # Extrair as caixas delimitadoras, os rótulos e os escores da saída do modelo
    boxes = output[0]['boxes'].int().numpy()
    labels = output[0]['labels'].numpy()
    scores = output[0]['scores'].numpy()

    # Desenhar as caixas delimitadoras na imagem original
    # Se houver pessoa preenche x
    image_with_boxes, x = service.untransform_and_draw_boxes(image_tensor[0], boxes, labels, scores)

    # salvar a imagem com as caixas delimitadoras
    # cv2.imshow('Image with boxes',  image_with_boxes)
    cv2.imwrite('Image_with_boxes.jpeg', image_with_boxes, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #save the models
    torch.save(model, 'model.pt')
    torch.save(model_local, 'model_local.pt')
