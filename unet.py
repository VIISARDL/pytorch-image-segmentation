import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class Ublock(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
		super().__init__()
		self.net = torch.nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size,padding=padding),
			nn.InstanceNorm2d(out_channels),
			#nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size,padding=padding),
			nn.InstanceNorm2d(out_channels),
			#nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)
	def forward(self, x):
		return self.net(x)

class UpSamplingPadding(torch.nn.Module):
	def __init__(self, in_channels, out_channels, bilinear=True):
		super(UpSamplingPadding, self).__init__()
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.ublock = Ublock(in_channels=self.in_channels, out_channels=self.out_channels)
	
	def forward(self, x1, x2):
		x1 = self.up(x1)
	   
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
						diffY // 2, diffY - diffY//2))
		
		# for padding issues, see 
		# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
		# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

		x = torch.cat([x2, x1], dim=1)
		x = self.ublock(x)
		return x


class Unet(nn.Module):
	def __init__(self, input_channels=3,input_width=480, input_height=360, n_classes=11):
		super(Unet,self).__init__()
		self.input_channels = input_channels
		self.input_width = input_width
		self.input_height = input_height
		self.n_classes = n_classes

		self.conv1 = Ublock(input_channels, 64, kernel_size=3)
		
		self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
		self.conv2 = Ublock(64, 128, kernel_size=3)
		
		self.pool3 = torch.nn.MaxPool2d(kernel_size=2)
		self.conv3 = Ublock(128, 256, kernel_size=3)

		self.pool4 = torch.nn.MaxPool2d(kernel_size=2)
		self.conv4 = Ublock(256, 512, kernel_size=3)

		self.pool5 = torch.nn.MaxPool2d(kernel_size=2)
		self.conv5 = Ublock(512, 512, kernel_size=3)

		self.up1 = UpSamplingPadding(512 + 512, 256)
		self.up2 = UpSamplingPadding(256 + 256, 128)
		self.up3 = UpSamplingPadding(128 + 128, 64)
		self.up4 = UpSamplingPadding(128, 64)

		self.outputconv = torch.nn.Conv2d(64, self.n_classes, kernel_size=1)


		# test weight init
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
				if m.bias is not None:
					m.bias.data.zero_()
	
		
	def forward(self,x):
		# Downsampling phase
		conv1 = self.conv1(x)
		
		pool2 = self.pool2(conv1)
		conv2 = self.conv2(pool2)
		
		pool3 = self.pool3(conv2)
		conv3 = self.conv3(pool3)

		pool4 = self.pool4(conv3)
		conv4 = self.conv4(pool4)

		pool5 = self.pool5(conv4)
		conv5 = self.conv5(pool5)

		# Upsampling phase
		up1 = self.up1(conv5,conv4)
		up2 = self.up2(up1,conv3)
		up3 = self.up3(up2,conv2)
		up4 = self.up4(up3,conv1)

		return F.sigmoid(self.outputconv(up4))


if __name__ == "__main__":
	import numpy as np
	batch_size = 1
	n_channels = 3
	input_width = 480
	input_height = 360
	n_classes = 10
	nz = torch.Tensor(np.zeros((batch_size,n_channels,input_width,input_height)))
	uz = torch.ones(batch_size,input_width*input_height,dtype=torch.long)
	model = Unet()
	outputs = model.forward(nz)
	
	criterion = nn.CrossEntropyLoss()
	learning_rate = 1e-4
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	
	print(outputs.shape, uz.shape)


	loss = criterion(outputs, uz)
	loss.backward()

	'''
	for t in range(500):
		# Forward pass: compute predicted y by passing x to the model.
		y_pred = model(x)

		# Compute and print loss.
		loss = loss_fn(y_pred, y)
		print(t, loss.item())

		# Before the backward pass, use the optimizer object to zero all of the
		# gradients for the variables it will update (which are the learnable
		# weights of the model). This is because by default, gradients are
		# accumulated in buffers( i.e, not overwritten) whenever .backward()
		# is called. Checkout docs of torch.autograd.backward for more details.
		optimizer.zero_grad()

		# Backward pass: compute gradient of the loss with respect to model
		# parameters
		loss.backward()

		# Calling the step function on an Optimizer makes an update to its
		# parameters
		optimizer.step()

	'''




	'''
	import hiddenlayer as hl
	hl_graph = hl.build_graph(model, nz)
	hl_graph.save("xxx", format="png")
	'''