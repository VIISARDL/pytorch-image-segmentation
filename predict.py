import argparse
import utils
from unet import Unet
#from segnet import Segnet
import torch
import numpy as np
#from utils import imageSegmentationGenerator
import torch.nn as nn
from dataset import MyCustomDataset
import random
import cv2


images_path = "dataset/dataset1/images_prepped_train/"
segs_path = "dataset/dataset1/annotations_prepped_train/"
batch_size = 1
n_classes = 12
input_height = 360#int(200)
input_width = 480#int(200)
input_channels = 3
custom_dataset_eval = MyCustomDataset(images_path=images_path, segs_path=segs_path, n_classes=n_classes, input_width=input_width,input_height=input_height)
custom_dataloader_eval = torch.utils.data.DataLoader(dataset=custom_dataset_eval,
												batch_size=batch_size,
												shuffle=False)


use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
model = Unet(input_channels=input_channels,input_width=input_width, input_height=input_height, n_classes=n_classes).to(device)
colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes+1)  ]


import os

temp = os.listdir()
model_list = []
for aux in temp:
	if(aux.endswith(".pth")):
		model_list.append(aux)



for model_path in model_list:
	model_path = './model_second.pth'
	model.load_state_dict(torch.load(model_path))
	model.eval() 

	for batch_idx, (data, target) in enumerate(custom_dataloader_eval):
		# get the inputs
		data, target = data.to(device).float(), target.to(device)

		# forward + backward + optimize
		predicts = model(data)
		
		#print(predicts.shape)
		#print(predicts)



		predicts = (predicts.view(-1,input_height,input_width))#.cpu().detach().numpy()    
		data = data.view(-1,360,480).cpu().detach().permute(1,2,0).numpy()
		
		seg = predicts#(predicts.view(input_height,input_width)).cpu().detach().numpy()    
		seg_img = np.zeros((input_height,input_width,input_channels))

		values, indices = torch.max(predicts,0)
		#print(values,indices)
		#values, indices = torch.max(predicts,1)
		#print(values,indices)
		#values, indices = torch.max(predicts,2)
		#print(values,indices)
		#print(predicts.shape)
		#print(predicts.argmax())
		#print(predicts[0][0][0])
		#print(predicts[1][0][0])
		
		indices = indices.cpu().detach().numpy()
		#print(type(indices))
		#print(indices.any() == np.float32(1))
		#exit(0)


		for c in range(n_classes+1):
			seg_img[:,:,0] += ((indices == np.float32(c))*( colors[int(c)][0] ))
			seg_img[:,:,1] += ((indices == np.float32(c))*( colors[int(c)][1] ))
			seg_img[:,:,2] += ((indices == np.float32(c))*( colors[int(c)][2] ))
			#seg_img[:,:,0] += ((seg[:,:] == np.float32(c))*( colors[int(c)][0] ))
			#seg_img[:,:,1] += ((seg[:,:] == np.float32(c))*( colors[int(c)][1] ))
			#seg_img[:,:,2] += ((seg[:,:] == np.float32(c))*( colors[int(c)][2] ))


		#cv2.imshow("predicts" , predicts)
		cv2.imshow("data" , data)
		cv2.imshow("seg_img" , seg_img)#seg_img) 
		cv2.imshow("seg_img1" , seg_img/255)#seg_img) 
		#
		#print(model_path.split("."))
		#print("./output/"+ model_path.split(".")[0] + str(".png"))
		#exit(0)
		#cv2.imwrite("./output/"+ model_path.split(".")[0] + str(".png"),seg_img)
		#cv2.imshow("predicts1" , predicts/255)
		#cv2.imshow("seg_img1" , seg_img/255)
		#cv2.imshow("predicts2" , predicts*255)
		#cv2.imshow("seg_img2" , seg_img*255)
		cv2.waitKey(30)
		#break
	break