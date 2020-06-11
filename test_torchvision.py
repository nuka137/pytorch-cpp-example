import torchvision
from torchsummary import summary

model = torchvision.models.resnet50()
summary(model, (3, 224, 224))
