import os
import torch
import torchvision
from torchvision import  transforms
from torch.utils.data import DataLoader
from PIL import Image


BATCH_SIZE = 8

images_dir = './images_dataset'
images_path = [os.path.join(images_dir, image_name) for image_name in  os.listdir(images_dir)]


def toTensor(images_path , to_size = (64,129)):
    full_tensor = []
    input_transformer = transforms.Compose([transforms.Resize((to_size[0], to_size[0]))])
    output_transformer = transforms.Compose([transforms.Resize((to_size[1], to_size[1]))])
    for image in images_path :
        image = torchvision.io.read_image(image).type(torch.float32) / 255.
        input_ = input_transformer(image)
        output_ = output_transformer(image)
        full_tensor.append((input_, output_))

    return full_tensor


# 64x64 -> 129x129
train_data = toTensor(images_path, to_size=(64,129))
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)