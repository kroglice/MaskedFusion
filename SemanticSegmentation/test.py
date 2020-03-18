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

from data_controller import SegDataset
from loss import Loss
from segnet import SegNet as segnet
import sys
sys.path.append("..")

import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/home/k7/YCB_Video_Dataset/', help="dataset root dir (''YCB_Video Dataset'')")
#parser.add_argument('--batch_size', default=2, help="batch size")
#parser.add_argument('--n_epochs', default=300, help="epochs to train")
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
#parser.add_argument('--lr', default=0.0001, help="learning rate")
#parser.add_argument('--logs_path', default='logs/', help="path to save logs")
parser.add_argument('--model_save_path', default='trained_models/', help="path to load models")
#parser.add_argument('--log_dir', default='logs/', help="path to save logs")
#parser.add_argument('--resume_model', default='', help="resume model name")
parser.add_argument('--img_path', default='outputs/', help="path to save images")
opt = parser.parse_args()

if __name__ == '__main__':
    test_dataset = SegDataset(opt.dataset_root, '{0}/{1}'.format(opt.dataset_root, 'dataset_config/test_data_list.txt'), False, 1000)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=int(opt.workers))

    print("Images on test_dataset", len(test_dataset))

    model = segnet()
    model.load_state_dict(torch.load(opt.model_save_path))
    model = model.cuda()
    model.eval()

    for j, data in enumerate(test_dataloader, 0):
        #print("Image {0}".format(j))
        #rgb, target = data  # Normal
        img_path, orig, rgb, target = data  # To test and see orig image
        rgb, target = Variable(rgb).cuda(), Variable(target).cuda()

        #print('img_path', img_path)

        start_time = time.time()

        semantic = model(rgb)

        print("--- %s seconds ---" % (time.time() - start_time))

        #''' Save orig image to see it
        orig = orig.detach().cpu().numpy()[0]
        orig = np.transpose(orig, (1, 2, 0))
        orig_img = Image.fromarray(orig.astype('uint8'), 'RGB')
        orig_img.save('{0}/rgb_{1}.png'.format(opt.img_path, j))
        #'''
        rgb = rgb.detach().cpu().numpy()[0]
        target = target.detach().cpu().numpy()[0]
        # convert output tensor to masked image
        seg_data = semantic[0]
        seg_data2 = torch.transpose(seg_data, 0, 2)
        seg_data2 = torch.transpose(seg_data2, 0, 1)
        seg_image = torch.argmax(seg_data2, dim=-1)
        obj_list = torch.unique(seg_image).detach().cpu().numpy()
        seg_image = seg_image.detach().cpu().numpy()

        #print("seg_image", np.unique(seg_image))
        image = seg_image.astype('uint8')
        # https://www.programcreek.com/python/example/89374/cv2.medianBlur
        medianblur = cv2.medianBlur(image, ksize=3)
        #print("medianblur", np.unique(medianblur))
        #gaussianblur = cv2.GaussianBlur(medianblur, ksize=(3, 3), sigmaX=0)
        #print("gaussianblur", np.unique(gaussianblur))
        # https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/
        #dillate = cv2.dilate(gaussianblur, kernel=(5, 5))
        dillate = cv2.dilate(medianblur, kernel=(5, 5))
        #print("dillate", np.unique(dillate))
        dillate_image = Image.fromarray(dillate.astype('uint8'))
        dillate_image.save('{0}/dillate_{1}.png'.format(opt.img_path, j))

        #'''
        seg_image = Image.fromarray(seg_image.astype('uint8'))
        seg_image.save('{0}/semantic_{1}.png'.format(opt.img_path, j))
        #'''
        #print("target", np.unique(target))
        target = Image.fromarray(target.astype('uint8'))
        target.save('{0}/target_{1}.png'.format(opt.img_path, j))



        my_mask = copy.deepcopy(dillate)
        my_mask[my_mask>0] = 255
        my_mask = cv2.cvtColor(my_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        #print("my_mask", np.unique(my_mask))
        res = cv2.bitwise_and(my_mask, orig)

        res = Image.fromarray(res)
        res.save('{0}/res_{1}.png'.format(opt.img_path, j))

        print("Labels", np.unique(dillate))
        print("{0} - {1} - saved...".format(j, img_path[0]))
        input("...")
    print('Test Finish')
