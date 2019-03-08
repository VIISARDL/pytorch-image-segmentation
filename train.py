import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import Unet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
import lovasz_losses as L
from torch.optim import lr_scheduler
from dataset import MyCustomDataset



def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def train_net(net,
			  epochs=100,
			  batch_size=4,
			  lr=0.1,
			  val_percent=0.05,
			  save_cp=True,
			  gpu=False,
			  img_scale=0.5):

	dir_img = 'data/train/'
	dir_mask = 'data/train_masks/'
	dir_checkpoint = 'checkpoints/'

	ids = get_ids(dir_img)
	ids = split_ids(ids)

	iddataset = split_train_val(ids, val_percent)

	images_path = "dataset/dataset1/images_prepped_train/"
	segs_path = "dataset/dataset1/annotations_prepped_train/"
	batch_size = 3
	n_classes = 12
	input_height = 360#int(200)
	input_width = 480#int(200)
	input_channels = 3



	print('''
	Starting training:
		Epochs: {}
		Batch size: {}
		Learning rate: {}
		Training size: {}
		Validation size: {}
		Checkpoints: {}
		CUDA: {}
	'''.format(epochs, batch_size, lr, len(iddataset['train']),
			   len(iddataset['val']), str(save_cp), str(gpu)))

	N_train = len(iddataset['train'])

	
	optimizer = optim.SGD(net.parameters(),
						  lr=lr,
						  momentum=0.99,
						  weight_decay=0.0005)
	
	#optimizer = optim.Adam(net.parameters(), lr=0.005)
	#criterion = nn.BCELoss()
	
	weights = [0.58872014284134,
						  0.51052379608154,
						  2.6966278553009,
						  0.45021694898605,
						  1.1785038709641,
						  0.77028578519821,
						  2.4782588481903,
						  2.5273461341858,
						  1.0122526884079,
						  3.2375309467316,
						  4.1312313079834,
						  0]


	CAMVID_MEAN = [0.41189489566336, 0.4251328133025, 0.4326707089857]
	CAMVID_STD = [0.27413549931506, 0.28506257482912, 0.28284674400252]

	#weights = [0.21085201899788072, 0.23258652172267635, 0.009829274523160764, 0.316582147542638, 0.04486270372893329, 0.09724054521142396, 0.011729535649409628, 0.011268086461802402, 0.0586568555101423, 0.006392310651932587]
	class_weights = torch.FloatTensor(weights).cuda()#1/torch.FloatTensor(weights).cuda()
	criterion = nn.CrossEntropyLoss(weight=class_weights)

	#scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
	#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, threshold=0.01, patience=5)#optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)



	custom_dataset = MyCustomDataset(images_path=images_path, segs_path=segs_path, n_classes=n_classes, input_width=input_width,input_height=input_height)
	custom_dataloader = torch.utils.data.DataLoader(dataset=custom_dataset,
													batch_size=batch_size,
													shuffle=True)



	for epoch in range(epochs):
		print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
		
		net.train()

		# reset the generators
		train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
		val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

		epoch_loss = 0

		for batch_idx, (data, target) in enumerate(custom_dataloader):

			data, target = data.cuda().float(), target.cuda()

			
			# forward + backward + optimize
			masks_pred = net(data)

			#print(masks_pred.shape, target.shape)

			loss = criterion(masks_pred,target)

			# print statistics
			epoch_loss += loss.item()

			#print('{0:.4f} --- loss: {1:.6f} --- lr: '.format(batch_idx * batch_size / len(custom_dataloader), loss.item()), get_lr(optimizer))
			print("Epoch (%d/%d) -  Batch (%5d/%5d) loss: %.3f (lr=%f)" % (epoch + 1,epochs, batch_idx + 1, len(custom_dataloader), loss.item(), get_lr(optimizer)))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			

		print('Epoch finished ! Loss: {}'.format(epoch_loss / batch_idx))
		
		#if 1:
		val_dice = eval_net(net, val, gpu)
		print('Validation Loss: {}'.format(val_dice))
		scheduler.step(val_dice)
		epoch_loss = 0
		if save_cp:
			torch.save(net.state_dict(),
					   dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
			print('Checkpoint {} saved !'.format(epoch + 1))

		
		



def get_args():
	parser = OptionParser()
	parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int',
					  help='number of epochs')
	parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
					  type='int', help='batch size')
	parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
					  type='float', help='learning rate')
	parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
					  default=True, help='use cuda')
	parser.add_option('-c', '--load', dest='load',
					  default=False, help='load file model')
	parser.add_option('-s', '--scale', dest='scale', type='float',
					  default=0.5, help='downscaling factor of the images')

	(options, args) = parser.parse_args()
	return options

if __name__ == '__main__':
	args = get_args()

	net = Unet(input_channels=3,input_width=480, input_height=360, n_classes=12)#Unet(n_channels=3, n_classes=1)

	if args.load:
		net.load_state_dict(torch.load(args.load))
		print('Model loaded from {}'.format(args.load))

	if args.gpu:
		net.cuda()
		# cudnn.benchmark = True # faster convolutions, but more memory

	try:
		train_net(net=net,
				  epochs=args.epochs,
				  batch_size=args.batchsize,
				  lr=args.lr,
				  gpu=args.gpu,
				  img_scale=args.scale)
	except KeyboardInterrupt:
		torch.save(net.state_dict(), 'INTERRUPTED.pth')
		print('Saved interrupt')
		try:
			sys.exit(0)
		except SystemExit:
			os._exit(0)
