import torch
from torch.utils.data.dataset import Dataset
import utils

import glob
import numpy as np
import cv2
from PIL import Image
import random
import argparse
import itertools

def getImageArr( path , width , height , imgNorm="divide" , odering='channels_first' ):

	try:
		img = cv2.imread(path, 1)
		debug = img.copy()
		#if imgNorm == "sub_and_divide":
		#	img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
		#elif imgNorm == "sub_mean":
		#	img = cv2.resize(img, ( width , height ))
		#	img = img.astype(np.float32)
		#	img[:,:,0] -= 103.939
		#	img[:,:,1] -= 116.779
		#	img[:,:,2] -= 123.68
		if imgNorm == "divide":
			img = cv2.resize(img, ( width , height ))
			img = img.astype(np.float32)
			img = img/255.0

		if odering == 'channels_first':
			img = np.rollaxis(img, 2, 0)
		return img, debug
	except Exception as e:
		print (path , e)
		img = np.zeros((  height , width  , 3 ))
		if odering == 'channels_first':
			img = np.rollaxis(img, 2, 0)
		return img, debug


def getSegmentationArr( path , nClasses ,  width , height  ):

	seg_labels = np.zeros((  height , width  , nClasses ))
	try:
		img = cv2.imread(path, 1)
		img = cv2.resize(img, ( width , height ))
		img = img[:, : , 0]
		debug = img.copy()
		for c in range(nClasses):
			seg_labels[: , : , c ] = (img == c ).astype(int)

	except Exception as e:
		print (e)
		
	seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
	return seg_labels,debug

		
class MyCustomDataset(Dataset):
	def __init__(self, images_path="dataset/dataset1/images_prepped_train/", segs_path="dataset/dataset1/annotations_prepped_train/",
				 n_classes=10, input_width=480,input_height=360, transforms=None):
		
		self.images_path = images_path
		self.segs_path = segs_path
		self.n_classes = n_classes
		self.input_width = input_width
		self.input_height = input_height
		self.transforms = transforms

		assert self.images_path[-1] == '/'
		assert self.segs_path[-1] == '/'
	
		images = glob.glob(self.images_path + "*.jpg") + glob.glob(self.images_path + "*.png") +  glob.glob(self.images_path + "*.jpeg")
		images.sort()
		segmentations = glob.glob(self.segs_path + "*.jpg") + glob.glob(self.segs_path + "*.png") +  glob.glob(self.segs_path + "*.jpeg")
		segmentations.sort()

		assert len( images ) == len(segmentations)

		X = []
		Y = []
		for img,seg in (zip(images,segmentations)):
			assert(  img.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0])
			img, debug = getImageArr(img , input_width , input_height )
			label, debug2 = getSegmentationArr( seg , n_classes , input_width , input_height )
			X.append(img)
			Y.append(np.argmax(np.array(label),axis=1))

		self.imgs = np.array(X)
		self.labels = np.array(Y)

	def __getitem__(self, index):
		return (self.imgs[index], self.labels[index])

	def __len__(self):
		return len(self.imgs)



if __name__ == "__main__":	
	images_path = "dataset/dataset1/images_prepped_train/"
	segs_path = "dataset/dataset1/annotations_prepped_train/"
	batch_size = 1
	n_classes = 10
	input_width = 480
	input_height = 360
	custom_dataset = MyCustomDataset(images_path=images_path, segs_path=segs_path, n_classes=n_classes, input_width=input_width,input_height=input_height)

	a,b = custom_dataset.__getitem__(5)
	print(a.shape,b.shape)