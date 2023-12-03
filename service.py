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
import albumentations as A
import random

LR = 0.00035
SAMPLES_QUANTITY = 500

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
    # Redimensionar a imagem para maximo 600x600 pixels
    image = cv2.resize(image, (600, 600))
    # Converter a imagem em um tensor
    image = torch.from_numpy(image).float()
    # Transpor a imagem para o formato CxHxW
    image = image.permute(2, 0, 1)
    # Normalizar a imagem
    image = image / 255.0
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                 std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([normalize])

    image = transform(image)
    
    # Adicionar uma dimensão de lote
    #transformation = torchvision.transforms.ToTensor()
    #image = transformation(image)
    image = image.unsqueeze(0)
    
    return image

# Definir uma função para desenhar as caixas delimitadoras na imagem
def untransform_and_draw_boxes(tensor_image: torch.Tensor, boxes : tuple, labels : tuple, scores : tuple, threshold=0.5, save_as = "person"):
    image = tensor_image.clone().detach()
    original_image = tensor_image.clone().detach()
    # Converter a imagem em um array numpy
    image = image.numpy()
    original_image = original_image.numpy()
    # Transpor a imagem para o formato HxWxC
    image = image.transpose(1, 2, 0)
    original_image = original_image.transpose(1, 2, 0)
    # Desnormalizar a imagem
    image = image * 255.0
    original_image = original_image * 255.0
    # Iterar sobre as caixas, os rótulos e os escores
    are_there_persons = []
    for box, label, score in zip(boxes, labels, scores):
        # Verificar se o escore é maior que o limiar
        if score > threshold:
            # Extrair as coordenadas da caixa
            x1, y1, x2, y2 = box
            # Se for person separar a miniimagem
            if label == 1:
                Path(__file__).parent.joinpath("data").joinpath("detected").joinpath(save_as).mkdir(parents=True, exist_ok=True)
                if x2>x1 and y2>y1:
                    cv2.imwrite(f'{Path(__file__).parent.joinpath("data").joinpath("detected").joinpath(save_as).joinpath(f"{save_as}_{int(10000*score)}_{int(time.time()+random.randint(1000,9999))}.jpeg")}', original_image[y1:y2, x1:x2])
                else:
                    print(f"Invalid box: {box}")
                are_there_persons.append(box)
            # Desenhar a caixa na imagem
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Escrever o rótulo e o escore na imagem
            image = cv2.putText(image, f'{classes[label]}: {score:.2f}', (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image, are_there_persons

def train(model : torchvision.models.detection.ssd.SSD, file_dir ='C://Users//gmart//Projects//Ai//DontStarveGamerAi//data//person', epochs = 1):
    old_device = model.parameters().__next__().device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device != old_device:
        model.to(device)
        print(f"Running on {device}")
        print(f"Allocated memory: {torch.cuda.memory_allocated(device)}")
    augmentator = A.Compose([
            A.Blur(blur_limit=3, p=0.1),
            A.MotionBlur(blur_limit=3, p=0.1),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.ToGray(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(p=0.3),
            A.RandomGamma(p=0.3),
            A.HorizontalFlip(p=0.1)],
              bbox_params= A.BboxParams(format='pascal_voc', 
                                        label_fields=['labels'])
        )

    # Transforms
    resize = torchvision.transforms.Resize((600, 600), antialias=False)
    to_tensor = torchvision.transforms.ToTensor()
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = torchvision.transforms.Compose([to_tensor, resize, normalize])
    
    model.train()
    
    # Define the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, nesterov=True)
    
    for j in range(epochs):
        print(f"Epoch: {j+1} / {epochs}")
        ima = []
        target = []
        bbox = []
        with open(Path(__file__).parent.joinpath('data').joinpath("annotations").joinpath("list_bbox_celeba.txt"), "r" ) as txt_file:
            bbox = txt_file.readlines()
        txt_file.close()
        bbox = {x.split()[0]: x.split()[1:]  for x in bbox}
        bbox = {k: [int(v[0]), int(v[1]), int(v[0])+int(v[2]), int(v[1])+int(v[3])] for k, v in bbox.items() if k != "image_id" and k != "202599"} 
        samples = os.listdir(file_dir)
        random.shuffle(samples)
        samples = samples[0:SAMPLES_QUANTITY]
        for j, file in enumerate(samples):
            if j%100 == 0:
                print(f"Progress: {j} / {len(samples)} Next sample: {file} Bbox: {bbox.get(file, [0,0,0,0])}")
            ima_= cv2.imread(file_dir + '//' + file)
            original_dimension = ima_.shape 
            
            x1, y1, x2, y2 = int(original_dimension[0]*0.05), int(original_dimension[1]*0.05), int(original_dimension[0]*0.95), int(original_dimension[1]*0.95)
            
            if file[0:8] == "anotated":
                x1=int(file.split("-")[1])
                y1=int(file.split("-")[2])
                x2=int(file.split("-")[3])
                y2=int(file.split("-")[4])

            x1, y1, x2, y2 = bbox.get(file, [x1, y1, x2, y2])

            """ try:
                augmented_image = augmentator(image=ima_, bboxes=[[x1,y1,x2,y2]], labels=[1])
                x1, y1, x2, y2 = augmented_image["bboxes"][0]
            except: """
            augmented_image = {"image":ima_}

            new_image = trans(augmented_image["image"])
            new_dimension = new_image.shape

            # bbox validation
            x1, y1, x2, y2 = int(new_dimension[1]*x1/original_dimension[0]), int(new_dimension[2]*y1/original_dimension[1]), int(new_dimension[1]*x2/original_dimension[0]), int(new_dimension[2]*y2/original_dimension[1])
            if x2<=x1 or y2<=y1:
                print(f"Invalid box: {file}")
                continue

            # add to batch and send to gpu
            ima.append(new_image.to(device))
            
            target.append({"boxes":torch.tensor(data=[[x1,y1,x2,y2]]).to(device), 
                            "labels":torch.tensor(data=[1]).to(device)})
            # max_batch_size = 8
            if len(ima) == 8:
                optimizer.zero_grad()
                output = model(images = ima, targets = target) 
                losses = output.get('bbox_regression', None)
                
                if losses is not None:
                    loss = sum(loss for loss in output.values())
                    if not torch.isnan(loss).any():
                        loss.backward()
                        optimizer.step()
                ima = []
                target = []
        # residual
        if len(ima) > 0:
            optimizer.zero_grad()
            output = model(images = ima, targets = target) 
            losses = output.get('bbox_regression', None)
            if losses is not None:
                loss = sum(loss for loss in output.values())
                if not torch.isnan(loss).any():
                    loss.backward()
                    optimizer.step()
            ima = []
            target = []
        print(f"Loss: {loss}")
    model.to(old_device)
    model.eval()
    return model

if __name__ == '__main__':
    print("Service module")