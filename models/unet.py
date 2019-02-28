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
  

class Unet(nn.Module):
	def __init__(self, input_channels=3,input_width=480, input_height=360, n_classes=10):
		super(Unet,self).__init__()
		self.input_channels = input_channels
		self.input_width = input_width
		self.input_height = input_height
		self.n_classes = n_classes

		self.conv1 = Conv2dSame(input_channels, 32, kernel_size=3)
		self.conv1_1 = Conv2dSame(32, 32, kernel_size=3)
		self.conv2 = Conv2dSame(32, 64, kernel_size=3)
		self.conv2_2 = Conv2dSame(64, 64, kernel_size=3)
		self.conv3 = Conv2dSame(64, 128, kernel_size=3)
		self.conv3_3 = Conv2dSame(128, 128, kernel_size=3)
		self.conv4 = Conv2dSame(128 + 64, 64, kernel_size=3)
		self.conv4_4 = Conv2dSame(64, 64, kernel_size=3)
		self.conv5 = Conv2dSame(64 + 32, 32, kernel_size=3)
		self.conv5_5 = Conv2dSame(32, 32, kernel_size=3)
		self.conv6 = Conv2dSame(32, n_classes, kernel_size=1)

		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.dropout = nn.Dropout(p=0.2)
		self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
		self.softmax = nn.Softmax(dim=1)

	def forward(self,x):
		# Downsampling phase
		conv1 = F.relu(self.conv1(x))
		conv1 = self.dropout(conv1)
		conv1 = F.relu(self.conv1_1(conv1))
		pool1 = self.pool(conv1)
		
		conv2 = F.relu(self.conv2(pool1))
		conv2 = self.dropout(conv2)
		conv2 = F.relu(self.conv2_2(conv2))
		pool2 = self.pool(conv2)

		conv3 = F.relu(self.conv3(pool2))
		conv3 = self.dropout(conv3)
		conv3 = F.relu(self.conv3_3(conv3))

		# Upsampling phase
		up1 = torch.cat((self.upsampling(conv3),conv2),dim=1)
		
		conv4 = F.relu(self.conv4(up1))
		conv4 = self.dropout(conv4)
		conv4 = F.relu(self.conv4_4(conv4))

		up2 = torch.cat((self.upsampling(conv4),conv1),dim=1)

		conv5 = F.relu(self.conv5(up2))
		conv5 = self.dropout(conv5)
		conv5 = F.relu(self.conv5_5(conv5))

		conv6 = F.relu(self.conv6(conv5))

		output = conv6.view(-1,self.n_classes,self.input_width*self.input_height)		
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