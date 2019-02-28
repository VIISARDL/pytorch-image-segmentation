import argparse
import utils
from models.unet import Unet
import torch
import numpy as np
from utils import imageSegmentationGenerator
import torch.nn as nn
from dataset import MyCustomDataset
import random
import cv2

def argParser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_images", type=str, default="dataset/dataset1/images_prepped_train/")
	parser.add_argument("--train_annotations", type=str, default="dataset/dataset1/annotations_prepped_train/")
	parser.add_argument("--n_classes", type=int, default=10)
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
validate = args.validate
epochs = 30#args.epochs

images_path = "dataset/dataset1/images_prepped_train/"
segs_path = "dataset/dataset1/annotations_prepped_train/"
batch_size = 16#*4
n_classes = 10
input_width = int(200)
input_height = int(200)
input_channels = 3
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

model = Unet(input_channels=input_channels,input_width=input_width, input_height=input_height, n_classes=n_classes).to(device)

weights = [0.21085202, 0.23258652, 0.00982927 , 0.31658215, 0.0448627, 0.09724055, 0.01172954, 0.01126809, 0.05865686, 0.00639231]#[0.13097618, 0.25962482, 0.00564374, 0.28953383, 0.08719237, 0.16336204, 0.00892258, 0.03077999, 0.01745943, 0.00650503]
#[0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 9.6446, 1.8418, 0.6823, 6.2478, 7.3614, 0]
class_weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)

#criterion = nn.CrossEntropyLoss()
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(),learning_rate)#, lr=learning_rate)
lr=0.1
'''
optimizer = torch.optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=0.9,
                            weight_decay=1e-6)
'''
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

custom_dataset = MyCustomDataset(images_path=images_path, segs_path=segs_path, n_classes=n_classes, input_width=input_width,input_height=input_height)
custom_dataloader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)


# labels 0-9
model = model.train()
for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    scheduler.step()
    for batch_idx, (data, target) in enumerate(custom_dataloader):
        running_loss = 0.0
        # get the inputs
        data, target = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        predicts = model.forward(data).cuda()#to(device=device)
        #predicts.cuda()#.to(device=device)

        #print(predicts.shape,target.shape)

        #print(torch.unique(target))

        #exit(0)

        loss = criterion(predicts,target)
        loss.backward()
        optimizer.step()

        #print(target)
        #print("MAX: ", target.max(1))
        #exit(0)

        # print statistics
        running_loss += loss.item()
        
        if batch_idx % 10:    # print every 2000 mini-batches
            print("Epoch (%d/%d) -  Batch (%5d/%5d) loss: %.3f (lr=%f)" % (epoch + 1,epochs, batch_idx + 1, len(custom_dataloader), running_loss, optimizer.param_groups[0]['lr']))
            running_loss = 0.0
        
print('Finished Training')


images_path = "dataset/dataset1/images_prepped_train/"
segs_path = "dataset/dataset1/annotations_prepped_train/"
batch_size = 1
n_classes = 10
custom_dataset_eval = MyCustomDataset(images_path=images_path, segs_path=segs_path, n_classes=n_classes, input_width=input_width,input_height=input_height)
custom_dataloader_eval = torch.utils.data.DataLoader(dataset=custom_dataset_eval,
                                                batch_size=batch_size,
                                                shuffle=False)



use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
model = Unet(input_channels=input_channels,input_width=input_width, input_height=input_height, n_classes=n_classes).to(device)

colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

model.eval() 
for batch_idx, (data, target) in enumerate(custom_dataloader_eval):
    # get the inputs
    data, target = data.to(device), target.to(device)

    # forward + backward + optimize
    predicts = model.forward(data).cuda()#to(device=device)
    
    seg = predicts.argmax(1)[0].view(input_width,input_height)
    seg_img = np.zeros((input_width,input_height,input_channels))

    for c in range(n_classes):
        seg_img[:,:,0] += ((seg.to("cpu").data.numpy()[:,:] == c )*( colors[c][0] ))#.astype('uint8')
        seg_img[:,:,1] += ((seg.to("cpu").data.numpy()[:,:] == c )*( colors[c][1] ))#.astype('uint8')
        seg_img[:,:,2] += ((seg.to("cpu").data.numpy()[:,:] == c )*( colors[c][2] ))#.astype('uint8')



    #print(seg_img[i][j])
    cv2.imshow("seg_img" , seg_img/255 )
    cv2.waitKey(0)


'''

images_path = "dataset/dataset1/images_prepped_train/"
segs_path = "dataset/dataset1/annotations_prepped_train/"
batch_size = 1
n_classes = 10
input_width = 480
input_height = 360


custom_dataset_eval = MyCustomDataset(images_path=images_path, segs_path=segs_path, n_classes=n_classes, input_width=input_width,input_height=input_height)
custom_dataloader_eval = torch.utils.data.DataLoader(dataset=custom_dataset_eval,
                                                batch_size=batch_size,
                                                shuffle=False)




use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
model = Unet(input_channels=input_channels,input_width=input_width, input_height=input_height, n_classes=n_classes).to(device)

colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

model.eval() 
total = np.zeros((10))
for batch_idx, (data, target) in enumerate(custom_dataloader_eval):
    # get the inputs
    data, target = data.to(device), target.to(device)
    total += np.bincount(target.view(-1).numpy(),minlength=n_classes)

    # forward + backward + optimize
    #predicts = model.forward(data).cuda()#to(device=device)
        
print(total/total.sum())    

'''