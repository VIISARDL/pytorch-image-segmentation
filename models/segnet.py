# References https://github.com/Sayan98/pytorch-segnet/blob/master/src/model.py 
# Small segnet version
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
    def forward(self, x):
        return self.net(x)
  

class Segnet(nn.Module):
	def __init__(self, input_channels=3,input_width=480, input_height=360, n_classes=10):
		super(Segnet,self).__init__()
		self.input_channels = input_channels
		self.input_width = input_width
		self.input_height = input_height
		self.n_classes = n_classes

		## Encode
		self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
		self.conv1_bn = nn.BatchNorm2d(64)
		self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
		self.conv2_bn = nn.BatchNorm2d(128)
		self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
		self.conv3_bn = nn.BatchNorm2d(256)
		self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
		self.conv4_bn = nn.BatchNorm2d(512)
		self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		## Decode
		#self.up1 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
		self.decoder1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
		self.decoder1_bn = nn.BatchNorm2d(256)

		#self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
		self.decoder2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
		self.decoder2_bn = nn.BatchNorm2d(128)
		
		#self.up3 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
		self.decoder3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
		self.decoder3_bn = nn.BatchNorm2d(64)
		
		self.decoder4 = nn.ConvTranspose2d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1)
		self.decoder4_bn = nn.BatchNorm2d(n_classes)


		self.softmax = nn.Softmax(dim=1)

	def forward(self,x):

		# Encoder phase
		conv1 = self.conv1(x)
		conv1 = F.relu(self.conv1_bn(conv1))
		conv1, ind1 = self.pool1(conv1)
				
		conv2 = self.conv2(conv1)
		conv2 = F.relu(self.conv2_bn(conv2))
		conv2, ind2 = self.pool2(conv2)
		
		conv3 = self.conv3(conv2)
		conv3 = F.relu(self.conv3_bn(conv3))
		conv3, ind3 = self.pool3(conv3)

		conv4 = self.conv4(conv3)
		conv4 = F.relu(self.conv4_bn(conv4))
		conv4, ind4 = self.pool4(conv4)

		# Decoder phase
		decod1 = F.max_unpool2d(conv4, ind4, kernel_size=2, stride=2, output_size=conv3.size())
		decod1 = self.decoder1(decod1)
		decod1 = F.relu(self.decoder1_bn(decod1))

		decod2 = F.max_unpool2d(decod1, ind3, kernel_size=2, stride=2, output_size=conv2.size())
		decod2 = self.decoder2(decod2)
		decod2 = F.relu(self.decoder2_bn(decod2))

		decod3 = F.max_unpool2d(decod2, ind2, kernel_size=2, stride=2, output_size=conv1.size())
		decod3 = self.decoder3(decod3)
		decod3 = F.relu(self.decoder3_bn(decod3))


		decod4 = F.max_unpool2d(decod3, ind1, kernel_size=2, stride=2, output_size=x.size())
		decod4 = self.decoder4(decod4)
		decod4 = F.relu(self.decoder4_bn(decod4))


		output = decod4.view(-1,self.n_classes,self.input_width*self.input_height)		
		output = self.softmax(output)
		return output


if __name__ == "__main__":
	import numpy as np
	batch_size = 1
	n_channels = 3
	input_width = 480
	input_height = 360
	n_classes = 10
	nz = torch.Tensor(np.zeros((batch_size,n_channels,input_width,input_height)))
	uz = torch.ones(batch_size,input_width*input_height,dtype=torch.long)
	model = Segnet()
	outputs = model.forward(nz)
	
	criterion = nn.CrossEntropyLoss()
	learning_rate = 1e-4
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	
	loss = criterion(outputs, uz)
	loss.backward()