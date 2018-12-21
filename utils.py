import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import os
from PIL import Image
import numpy as np
import torch

def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    print(image_name)
    image = cv2.imread(image_name)
    image = cv2.resize(image,(512,512))#.transpose(2,0,1)
   # print('1', image.shape)
   # image = Image.open(image_name)
    print(image.size)
    #image =image.resize([512,512],Image.BILINEAR)
    #image = np.asarray(image)

   # image = Image.fromarray(np.uint8(image))
    #print('2',image.shape)

    image2 = Image.fromarray(image)
    image2 = Variable(loader(image2))
    print('[]', image2.shape)

    # fake batch dimension required to fit network's input dimensions
    image2 = image2.unsqueeze(0)
    print('++++',image2.shape)
    return image2

def image_loader_gray(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    image = Image.open(image_name).convert('L')
    image = image.resize([512,512])
    image = np.asarray(image)
    image = np.asarray([image,image,image])
    image = Image.fromarray(np.uint8(image).transpose(1,2,0))
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    print('---',image.shape)
    return image

def save_image(tensor, size, input_size, fname='transferred.png'):
    unloader = transforms.ToPILImage()  # reconvert into PIL image

    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(size)
    image = unloader(image).resize(input_size)

    out_path = os.path.join('real2syn', fname)
    if not os.path.exists('real2syn'):
        os.mkdir('real2syn')

    image.save(out_path)
