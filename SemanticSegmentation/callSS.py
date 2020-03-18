import os
import copy
import random
import argparse
import time
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn

from SemanticSegmentation.loss import Loss
from SemanticSegmentation.segnet import SegNet as segnet
import sys
sys.path.append("..")

import cv2
from torchvision import transforms

def execSS(rgb):

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    model = segnet()
    model.load_state_dict(torch.load("SemanticSegmentation/trained_models/model_97_0.11637805987335742.pth"))
    model = model.cuda()
    model.eval()

    #print('rgb', type(rgb))
    #print('rgb', rgb.size)
    rgb = np.asarray(rgb)
    #print('rgb', rgb.shape)
    rgb = np.transpose(rgb, (2, 0, 1))
    #print('rgb', rgb.shape)
    #print('rgb', type(rgb))
    rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
    #print('rgb', rgb.shape)
    #print('rgb', type(rgb))
    rgb = rgb.unsqueeze(0)
    #print('rgb', rgb.shape)
    #print('rgb', type(rgb))

    rgb = Variable(rgb).cuda()
    semantic = model(rgb)
    #print('semantic', semantic.shape)

    # convert output tensor to masked image
    seg_data = semantic[0]
    seg_data2 = torch.transpose(seg_data, 0, 2)
    seg_data2 = torch.transpose(seg_data2, 0, 1)
    seg_image = torch.argmax(seg_data2, dim=-1)
    obj_list = torch.unique(seg_image).detach().cpu().numpy()
    seg_image = seg_image.detach().cpu().numpy()
    #print('seg_image', seg_image.shape)

    image = seg_image.astype('uint8')
    #print('image', image.shape)
    medianblur = cv2.medianBlur(image, ksize=3)
    #print('medianblur', medianblur.shape)
    #dillate = cv2.dilate(medianblur, kernel=(5, 5))
    dillate = cv2.dilate(medianblur, kernel=(7, 7))
    #print('dillate', dillate.shape)
    dillate_image = Image.fromarray(dillate.astype('uint8'))
    #print('dillate_image', dillate_image.size)

    return dillate_image
