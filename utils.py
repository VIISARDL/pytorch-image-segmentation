import glob
import numpy as np
import cv2
import random
import argparse
import itertools


'''
def imageSegmentationGenerator( images_path , segs_path ,  n_classes ):

	assert images_path[-1] == '/'
	assert segs_path[-1] == '/'

	images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
	images.sort()
	segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
	segmentations.sort()

	colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

	assert len( images ) == len(segmentations)

	for im_fn , seg_fn in zip(images,segmentations):
		assert(  im_fn.split('/')[-1] ==  seg_fn.split('/')[-1] )

		img = cv2.imread( im_fn )
		seg = cv2.imread( seg_fn )
		print (np.unique( seg ))

		seg_img = np.zeros_like( seg )

		for c in range(n_classes):
			seg_img[:,:,0] += ( (seg[:,:,0] == c )*( colors[c][0] )).astype('uint8')
			seg_img[:,:,1] += ((seg[:,:,0] == c )*( colors[c][1] )).astype('uint8')
			seg_img[:,:,2] += ((seg[:,:,0] == c )*( colors[c][2] )).astype('uint8')

		cv2.imshow("img" , img )
		cv2.imshow("seg_img" , seg_img )
		cv2.waitKey()
'''

def getImageArr( path , width , height , imgNorm="sub_mean" , odering='channels_first' ):

	try:
		img = cv2.imread(path, 1)
		debug = img.copy()
		if imgNorm == "sub_and_divide":
			img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
		elif imgNorm == "sub_mean":
			img = cv2.resize(img, ( width , height ))
			img = img.astype(np.float32)
			img[:,:,0] -= 103.939
			img[:,:,1] -= 116.779
			img[:,:,2] -= 123.68
		elif imgNorm == "divide":
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

	print(seg_labels.shape)
	exit(0)
	return seg_labels,debug


def imageSegmentationGenerator(images_path, segs_path, batch_size, n_classes, input_height, input_width):
	assert images_path[-1] == '/'
	assert segs_path[-1] == '/'
	
	images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
	images.sort()
	segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
	segmentations.sort()

	assert len( images ) == len(segmentations)
	for im , seg in zip(images,segmentations):
		assert(  im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )


	zipped = itertools.cycle( zip(images,segmentations) )

	while True:
		X = []
		Y = []
		for _ in range( batch_size) :
			im , seg = next(zipped)
			img, debug = getImageArr(im , input_width , input_height )
			label, debug2 = getSegmentationArr( seg , n_classes , input_width , input_height )
			X.append(img)
			Y.append(label)
		#cv2.imshow("teste",np.asarray(debug).astype('uint8')/255.0)
		#cv2.waitKey(30)
		#cv2.imshow("teste2",np.asarray(debug2).astype('uint8')*255.0)
		#cv2.waitKey(30)
		yield np.array(X) , np.array(Y)


if __name__ == "__main__":
	# import Models , LoadBatches
	# G  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_train/" ,  "data/clothes_seg/prepped/annotations_prepped_train/" ,  1,  10 , 800 , 550 , 400 , 272   ) 
	# G2  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_test/" ,  "data/clothes_seg/prepped/annotations_prepped_test/" ,  1,  10 , 800 , 550 , 400 , 272   ) 

	# m = Models.VGGSegnet.VGGSegnet( 10  , use_vgg_weights=True ,  optimizer='adadelta' , input_image_size=( 800 , 550 )  )
	# m.fit_generator( G , 512  , nb_epoch=10 )

	#parser = argparse.ArgumentParser()
	#parser.add_argument("--images", type = str  )
	#parser.add_argument("--annotations", type = str  )
	#parser.add_argument("--n_classes", type=int )
	#args = parser.parse_args()
	
	images_path = "dataset/dataset1/images_prepped_test/"
	segs_path = "dataset/dataset1/annotations_prepped_test/"
	batch_size = 1
	n_classes = 10
	input_width = 480
	input_height = 360

	imageSegmentationGenerator(images_path, segs_path, batch_size, n_classes, input_height, input_width)
	