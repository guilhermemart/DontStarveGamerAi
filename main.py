import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import torchvision
import seaborn as sn
from PIL import Image
import cv2


def get_path(relpath):
  return os.path.join("c:/Users/guilh/napp/dontStarve/DontStarveGamerAi/", BASE_DIR, DATA_DIR, relpath)


def lookat_dataset(dataset, istensor=False):
  figure = plt.figure(figsize=(8, 8))
  rows, cols = 2, 2
  for i in range(1, 5):
      sample_idx = torch.randint(len(dataset), size=(1,)).item()
      img, label = dataset[sample_idx]
      figure.add_subplot(rows, cols, i)
      plt.title(CATEGORIES[label])
      plt.axis("off")
      if istensor:
        plt.imshow(img.squeeze().permute(1, 2, 0))
      else:
        plt.imshow(img)
  plt.show()


class MLPClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    
    self.layers = nn.Sequential(
        nn.Linear(3 * 32*32, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

  def forward(self, x):
    v = self.flatten(x)
    return self.layers(v)


class ConvolutionalModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.convlayers = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(16, 32, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

      )

      self.linearlayers = nn.Sequential(
          nn.Linear(1152, 256),
          nn.ReLU(),
          nn.Linear(256, 10)
      )

  def forward(self, x):
      x = self.convlayers(x)
      x = torch.flatten(x, 1)
      return self.linearlayers(x)
  
def train(model, dataloader, loss_func, optimizer):
  model.train()
  cumloss = 0.0

  for imgs, labels in dataloader:
    imgs, labels = imgs.to(device), labels.to(device)
    
    optimizer.zero_grad()

    pred = model(imgs)

    loss = loss_func(pred, labels)
    loss.backward()
    optimizer.step()

    cumloss += loss.item()

  return cumloss / len(dataloader)

def validate(model, dataloader, loss_func):
  model.eval()
  cumloss = 0.0

  with torch.no_grad():
    for imgs, labels in dataloader:
      imgs, labels = imgs.to(device), labels.to(device)

      pred = model(imgs)
      loss = loss_func(pred, labels)
      cumloss += loss.item()

  return cumloss / len(dataloader)

def plot_losses(losses):
  fig = plt.figure(figsize=(13, 5))
  ax = fig.gca()
  for loss_name, loss_values in losses.items():  
    ax.plot(loss_values, label=loss_name)
  ax.legend(fontsize="16")
  ax.set_xlabel("Iteration", fontsize="16")
  ax.set_ylabel("Loss", fontsize="16")
  ax.set_title("Loss vs iterations", fontsize="16")
  plt.show()

def make_confusion_matrix(model, loader, n_classes):
  confusion_matrix = torch.zeros(n_classes, n_classes, dtype=torch.int64)
  with torch.no_grad():
    for i, (imgs, labels) in enumerate(loader):
      imgs = imgs.to(device)
      labels = labels.to(device)
      outputs = model(imgs)
      _, predicted = torch.max(outputs, 1)
      for t, p in zip(torch.as_tensor(labels, dtype=torch.int64).view(-1), 
                      torch.as_tensor(predicted, dtype=torch.int64).view(-1)):
        confusion_matrix[t, p] += 1
  return confusion_matrix

def evaluate_accuracy(model, dataloader, classes, verbose=True):
  # prepare to count predictions for each class
  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}

  confusion_matrix = make_confusion_matrix(model, dataloader, len(classes))
  if verbose:
    total_correct = 0.0
    total_prediction = 0.0
    for i, classname in enumerate(classes):
      correct_count = confusion_matrix[i][i].item()
      class_pred = torch.sum(confusion_matrix[i]).item()

      total_correct += correct_count
      total_prediction += class_pred

      accuracy = 100 * float(correct_count) / class_pred
      print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                    accuracy))
  print("Global acccuracy is {:.1f}".format(100 * total_correct/total_prediction))
  return confusion_matrix

def test(model, dataloader, classes):
  # prepare to count predictions for each class
  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}

  # again no gradients needed
  with torch.no_grad():
      for images, labels in dataloader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          _, predictions = torch.max(outputs, 1)
          # collect the correct predictions for each class
          for label, prediction in zip(labels, predictions):
              if label == prediction:
                  correct_pred[classes[label]] += 1
              total_pred[classes[label]] += 1

  # print accuracy for each class
  total_correct = 0.0
  total_prediction = 0.0
  for classname, correct_count in correct_pred.items():
      total_correct += correct_count
      total_prediction += total_pred[classname]
      accuracy = 100 * float(correct_count) / total_pred[classname]
      print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                    accuracy))
  print("Global acccuracy is {:.1f}".format(100 * total_correct/total_prediction))

BASE_DIR = '/datasets'
DATA_DIR = '/cifar10'
CATEGORIES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

prep_transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                    )
                  ])

# Downloading and Applying a transform
tensor_train = CIFAR10(get_path("train"), train=True, download=False,
                         transform=prep_transform)
tensor_test = CIFAR10(get_path("test"), train=False, download=False,
                         transform=prep_transform)

imgs = torch.stack([img_t for img_t, _ in tensor_train], dim=3)
imgs.shape
     
imgs.view(3, -1).mean(dim=1)
     
imgs.view(3, -1).std(dim=1)

batch_size = 64
train_loader = DataLoader(tensor_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(tensor_test, batch_size=batch_size, shuffle=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Rodando na {device}")


# Linear model
model = MLPClassifier().to(device)
if os.path.exists("c:/Users/guilh/napp/dontStarve/DontStarveGamerAi/datasets/cifar10/saves/model.pt"):
  model.load_state_dict(torch.load("c:/Users/guilh/napp/dontStarve/DontStarveGamerAi/datasets/cifar10/saves/model.pt"))
  print("Modelo carregado")


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 2
train_losses = []
test_losses = []
for t in range(epochs):
  train_loss = train(model, train_loader, loss_func, optimizer)
  train_losses.append(train_loss)
  if t % 10 == 0:
    print(f"Epoch: {t}; Train Loss: {train_loss}")
  
  test_loss = validate(model, test_loader, loss_func)
  test_losses.append(test_loss)

losses = {"Train loss": train_losses, "Test loss": test_losses}
plot_losses(losses)



confusion_matrix = evaluate_accuracy(model, test_loader, CATEGORIES)
     

plt.figure(figsize=(12, 12))
sn.set(font_scale=1.4)
sn.heatmap(confusion_matrix.tolist(), 
           annot=True, annot_kws={"size": 16}, fmt='d')
plt.show()
     

# convolutional model
model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
convmodel = ConvolutionalModel().to(device)
if os.path.exists("c:/Users/guilh/napp/dontStarve/DontStarveGamerAi/datasets/cifar10/saves/convmodel.pt"):
  convmodel.load_state_dict(torch.load("c:/Users/guilh/napp/dontStarve/DontStarveGamerAi/datasets/cifar10/saves/convmodel.pt"))
  print("Modelo convolucional carregado")


loss_func2 = nn.CrossEntropyLoss()
optimizer2 = torch.optim.SGD(convmodel.parameters(), lr=0.001)
     

epochs = 1
conv_train_losses = []
conv_test_losses = []
for t in range(epochs):
  train_loss = train(convmodel, train_loader, loss_func2, optimizer2)
  conv_train_losses.append(train_loss)
  if t % 10 == 0:
    print(f"Epoch: {t}; Train Loss: {train_loss}")
  test_loss = validate(convmodel, test_loader, loss_func2)
  conv_test_losses.append(test_loss)  

conv_losses = {"Train Loss": conv_train_losses, "Test Loss": conv_test_losses}
plot_losses(conv_losses)

confusion_matrix = evaluate_accuracy(convmodel, test_loader, CATEGORIES)
     

plt.figure(figsize=(12, 12))
sn.set(font_scale=1.4)
sn.heatmap(confusion_matrix.tolist(), 
           annot=True, annot_kws={"size": 16}, fmt='d')
plt.show()

cap = cv2.VideoCapture(0)

prep_transforms = T.Compose(
    [T.Resize((32, 32)),
     T.ToTensor(),
     T.Normalize( (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616) )
     ]
)

convmodel.eval()

while True:
    ret, frame = cap.read()  # Captura um quadro da câmera

    # Transforme o quadro em uma imagem PyTorch
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = prep_transforms(img_pil)

    # Realize a inferência no modelo
    with torch.no_grad():
        batch = img_tensor.unsqueeze(0).to(device)
        output = convmodel(batch)  # Adicione uma dimensão para o lote (batch)

    # Processar a saída do modelo aqui (dependendo do seu modelo)
    logits = torch.nn.functional.softmax(output, dim=1) * 100
    x, y = output
    prob_dict = {}
    for i, classname in enumerate(CATEGORIES):
      prob = logits[0][i].item()
      print(f"{classname} score: {prob:.2f}")
      prob_dict[classname] = [prob]

    # Exibir a imagem de saída e aguardar a tecla 'q' para sair
    cv2.imshow('Capturando Tela', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

img = Image.open("c:/Users/guilh/napp/dontStarve/DontStarveGamerAi/datasets/localImages/carrof1.png")

img_tensor = prep_transforms(img)

""" 
plt.imshow(img_tensor.permute(1,2, 0))
plt.show() """

batch = img_tensor.unsqueeze(0).to(device)
     


output = convmodel(batch)

logits = torch.nn.functional.softmax(output, dim=1) * 100
prob_dict = {}
for i, classname in enumerate(CATEGORIES):
  prob = logits[0][i].item()
  print(f"{classname} score: {prob:.2f}")
  prob_dict[classname] = [prob]

torch.save(model.state_dict(), "c:/Users/guilh/napp/dontStarve/DontStarveGamerAi/datasets/cifar10/saves/model.pt")
torch.save(convmodel.state_dict(), "c:/Users/guilh/napp/dontStarve/DontStarveGamerAi/datasets/cifar10/saves/convmodel.pt")