import argparse
import utils
from models.unet import Unet
from models.segnet import Segnet
import torch
import numpy as np
#from utils import imageSegmentationGenerator
import torch.nn as nn
from dataset import MyCustomDataset
import random
import cv2

def argParser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_images", type=str, default="dataset/dataset1/images_prepped_train/")
	parser.add_argument("--train_annotations", type=str, default="dataset/dataset1/annotations_prepped_train/")
	parser.add_argument("--n_classes", type=int, default=12)
	parser.add_argument("--input_width", type=int , default=480)
	parser.add_argument("--input_height", type=int , default=360)
	parser.add_argument('--validate',action='store_false')
	parser.add_argument("--val_images", type = str , default="dataset/dataset1/images_prepped_test/")
	parser.add_argument("--val_annotations", type = str , default="dataset/dataset1/annotations_prepped_test/")
	parser.add_argument("--epochs", type = int, default=100)
	parser.add_argument("--batch_size", type = int, default=1)

	return parser.parse_args()

args = argParser()
train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

custom_dataset = MyCustomDataset(images_path=train_images_path, segs_path=train_segs_path, n_classes=n_classes, input_width=input_width,input_height=input_height)
custom_dataloader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                batch_size=train_batch_size,
                                                shuffle=False)



total = np.zeros((12))
for batch_idx, (data, target) in enumerate(custom_dataloader):
    # get the inputs
    data, target = data.to(device), target.to(device)
    total += np.bincount(target.view(-1).numpy(),minlength=n_classes)

ss ="["
aux = (total/total.sum())
for i in range(0,len(aux)):
    ss+=str(aux[i])
    if(i != len(aux)-1):
        ss+=(", ")
ss+="]"

print(ss)