from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models

import copy
import argparse

from loss import ContentLoss, GramMatrix, StyleLoss
from utils import image_loader, image_loader_gray, save_image

parser = argparse.ArgumentParser()
parser.add_argument('--content', '-c', type=str,default='contents/golden_gate.jpg', required=True, help='The path to the Content image')
parser.add_argument('--style', '-s', type=str, required=True, default='styles/woman-with-hat-matisse.jpg',help='The path to the style image')
parser.add_argument('--epoch', '-e', type=int, default=500, help='The number of epoch')
parser.add_argument('--content_weight', '-c_w', type=int, default=1, help='The weight of content loss')
parser.add_argument('--style_weight', '-s_w', type=int, default=500, help='The weight of style loss')
parser.add_argument('--initialize_noise', '-i_n', action='store_true', help='Initialize with white noise? elif initialize with content image')
parser.add_argument('--cuda', action='store_true',default=True, help='use cuda?')
args = parser.parse_args()

use_cuda = torch.cuda.is_available() and args.cuda
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# desired size of the output image
imsize = 512 if use_cuda else 128  # use small size if no gpu


#assert style_img.size() == content_img.size(), \
#    "we need to import style and content images of the same size"

cnn = models.vgg19(pretrained=True).features

# move it to the GPU if possible:
if use_cuda:
    cnn = cnn.cuda()

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, style_img, content_img, style_weight=1000, content_weight=1, content_layers=content_layers_default, style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    return model, style_losses, content_losses

def get_input_param_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300, style_weight=1000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img, style_weight, content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(float(style_score), float(content_score)))

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_param.data.clamp_(0, 1)

    return input_param.data



#style_img = '/home/dut-ai/Documents/temp/synthia/test/RGB/SEQ6_left_0000000.png'


#
# content_img = '/home/dut-ai/Documents/temp/cityscape/Cityscape/leftImg8bit_trainvaltest/leftImg8bit/RGB_train/aachen_000006_000019_leftImg8bit.png'
#/home/dut-ai/Documents/temp/synthia/train/RGB/SEQ1_left_0000001.png
content_path = '/home/dut-ai/Documents/temp/cityscape/Cityscape/leftImg8bit_trainvaltest/leftImg8bit/RGB_train'
imgs = os.listdir(content_path)

for i,img in enumerate(imgs):
    print(i)
    img_path = os.path.join(content_path,img)

    style_img = image_loader(args.style, imsize).type(dtype)
    content_img = image_loader_gray(img_path, imsize).type(dtype)
#style_img = style_img.resize(640,169)
#content_img = content_img.resize(640,169)

    if args.initialize_noise:
        input_img = Variable(torch.randn(content_img.data.size())).type(dtype)
    else:
        input_img = image_loader_gray(args.content, imsize).type(dtype)

    input_size = Image.open(img_path).size

    output = run_style_transfer(cnn, content_img, style_img, input_img, args.epoch, args.style_weight, args.content_weight)

    name_content = img[:-3]
    name_style, _ = os.path.splitext(os.path.basename(args.style))
    fname = name_content+'-'+name_style+'.jpg'

    save_image(output, size=input_img.data.size()[1:], input_size=input_size, fname=fname)


