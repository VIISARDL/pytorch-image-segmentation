# pytorch-image-segmentation
Image Segmentation in Pytorch


## Models 

* FCN8				TO BE DONE
* FCN32				TO BE DONE
* Simple Segnet 	TO BE DONE
* VGG Segnet 		TO BE DONE
* U-Net 			TO BE DONE
* VGG U-Net 		TO BE DONE

## Getting Started

### Prerequisites

* Python 3.6.7
* Pytorch 1.0
* Opencv 3.4.2

### How to install dependencies


### Preparing the data for training

You need to make two folders

Images Folder - For all the training images
Annotations Folder - For the corresponding ground truth segmentation images
The filenames of the annotation images should be same as the filenames of the RGB images.

The size of the annotation image for the corresponding RGB image should be same.

For each pixel in the RGB image, the class label of that pixel in the annotation image would be the value of the blue pixel.

Example code to generate annotation images :


```python
import cv2
import numpy as np

ann_img = np.zeros((30,30,3)).astype('uint8')
ann_img[ 3 , 4 ] = 1 # this would set the label of pixel 3,4 as 1

cv2.imwrite( "ann_1.png" ,ann_img )
```

Only use bmp or png format for the annotation images.



References

https://github.com/divamgupta/image-segmentation-keras.git


Download the sample prepared dataset
Download and extract the following:

https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing

Place the dataset1/ folder in data/