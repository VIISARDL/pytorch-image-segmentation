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



	#weights = [0.21085201899788072, 0.23258652172267635, 0.009829274523160764, 0.316582147542638, 0.04486270372893329, 0.09724054521142396, 0.011729535649409628, 0.011268086461802402, 0.0586568555101423, 0.006392310651932587]
	class_weights = torch.FloatTensor(weights).cuda()#1/torch.FloatTensor(weights).cuda()
	criterion = nn.CrossEntropyLoss(weight=class_weights)

	#scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
	#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, threshold=0.01, patience=5)#optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

	


	for epoch in range(epochs):
		print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
		
		net.train()

		# reset the generators
		train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
		val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

		epoch_loss = 0

		for i, b in enumerate(batch(train, batch_size)):
			imgs = np.array([i[0] for i in b]).astype(np.float32)
			true_masks = np.array([i[1] for i in b])

			imgs = torch.from_numpy(imgs)
			true_masks = torch.from_numpy(true_masks)

			if gpu:
				imgs = imgs.cuda()
				true_masks = true_masks.cuda()

			masks_pred = net(imgs)

			#print(true_masks.max())
			#exit(0)
			#print(masks_pred.shape,true_masks.shape)
			masks_probs_flat = masks_pred#.view(-1)
			true_masks_flat = true_masks.long()#.view(-1)
			#true_masks_flat.scatter_(1, true_masks_flat.view(-1,1),1);

			#print(masks_probs_flat.shape,true_masks_flat.shape)
			loss = criterion(masks_probs_flat, true_masks_flat)
			#loss = L.lovasz_softmax(masks_probs_flat, true_masks_flat)
			epoch_loss += loss.item()

			print('{0:.4f} --- loss: {1:.6f} --- lr: '.format(i * batch_size / N_train, loss.item()), get_lr(optimizer))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
		print('Epoch finished ! Loss: {}'.format(epoch_loss / i))
		
		#if 1:
		val_dice = eval_net(net, val, gpu)
		print('Validation Loss: {}'.format(val_dice))
		scheduler.step(val_dice)
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
