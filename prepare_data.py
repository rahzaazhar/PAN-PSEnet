import os
import numpy as np 
import cv2
import lmdb

image_dir = '/home/azhar/summerResearch/data/IC19MLT_root/train_IC19MLT/'
image_gt_dir = '/home/azhar/summerResearch/data/IC19MLT_root/train_IC19MLT_gt/'
word_image_dir = '/home/azhar/crnn-pytorch/data/word_images/'
outputPath = '/home/azhar/crnn-pytorch/data/'
cnt = 3748


def crop(imgpath,bbox):
	img = cv2.imread(imgpath)
	bbox = bbox.reshape(4,2)
	topleft_x = np.min(bbox[:,0])
	topleft_y = np.min(bbox[:,1])
	bot_right_x = np.max(bbox[:,0])
	bot_right_y = np.max(bbox[:,1])
	cropped_img = img[topleft_y:bot_right_y, topleft_x:bot_right_x]
	return cropped_img
	

def checkImageIsValid(imageBin):
	if imageBin is None:
		return False
	imageBuf = np.fromstring(imageBin, dtype=np.uint8)
	if imageBuf.size == 0:
		return False
	img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
	imgH, imgW = img.shape[0], img.shape[1]
	if imgH * imgW == 0:
		return False
	return True

def writeCache(env, cache):
	with env.begin(write=True) as txn:
		for k, v in cache.items():
			if type(k)==str:
				k = k.encode()
			if type(v)==str:
				v = v.encode()
			txn.put(k, v)


def create_lists():
	global cnt
	imgPathList = []
	labelList = []
	if not os.path.exists(word_image_dir):
		os.mkdir(word_image_dir)

	gt_filelist = os.listdir(image_gt_dir)
	gt_filelist.sort()
	#gt_filelist = gt_filelist[0:9801]

	for gt_file in gt_filelist:
		gt_filename = gt_file.split('.')[0]
		k = int(gt_filename.split('_')[-1])
		if 9000<=k<=10000:
			f = open(image_gt_dir+gt_file,'r')
			lines = f.readlines()
			for line in lines:
				elements = line.split(',')
				if elements[-2]=='Hindi':
					bbox = [int(ele) for ele in elements[0:8]]
					bbox = np.array(bbox)
					label = elements[-1].strip()
					if cnt<10:
						print(label)
					imgpath = image_dir+gt_filename+'.png'
					try:
						cropped_img = crop(imgpath,bbox)
					except:
						print(".jpg image ignore")
						continue

					word_image_path = word_image_dir+"img"+str(cnt)+'.png'
					cv2.imwrite(word_image_path,cropped_img)
					imgPathList.append(word_image_path)
					labelList.append(label)
					cnt = cnt+1
	return imgPathList, labelList


def createLMDBDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
	"""
	Create LMDB dataset for CRNN training.
	ARGS:
		outputPath    : LMDB output path
		imagePathList : list of image path
		labelList     : list of corresponding groundtruth texts
		lexiconList   : (optional) list of lexicon lists
		checkValid    : if true, check the validity of every image
	"""
	assert(len(imagePathList) == len(labelList))
	nSamples = len(imagePathList)
	env = lmdb.open(outputPath, map_size=1099511627776)
	cache = {}
	cnt = 1
	for i in range(nSamples):
		imagePath = imagePathList[i]
		label = labelList[i]
		if not os.path.exists(imagePath):
			print('%s does not exist' % imagePath)
			continue
		with open(imagePath, 'rb') as f:
			imageBin = f.read()
		if checkValid:
			if not checkImageIsValid(imageBin):
				print('%s is not a valid image' % imagePath)
				continue

		imageKey = 'image-%09d' % cnt
		labelKey = 'label-%09d' % cnt
		cache[imageKey] = imageBin
		cache[labelKey] = label
		if lexiconList:
			lexiconKey = 'lexicon-%09d' % cnt
			cache[lexiconKey] = ' '.join(lexiconList[i])
		if cnt % 1000 == 0:
			writeCache(env, cache)
			cache = {}
			print('Written %d / %d' % (cnt, nSamples))
		cnt += 1
	nSamples = cnt-1
	cache['num-samples'] = str(nSamples)
	writeCache(env, cache)
	print('Created dataset with %d samples' % nSamples)

def generate_gt(outputPath,imagePathList,labelList):
	path = outputPath+'test_gt.txt'
	with open(path,'w') as f:
		for image,label in zip(imagePathList,labelList):
			line = image+','+label+'\n'
			f.write(line)


def main():
	imgPathList, labelList = create_lists()
	print(len(imgPathList),len(labelList))
	#createDataset(outputPath,imgPathList,labelList)
	generate_gt(outputPath,imgPathList,labelList)

if __name__ == '__main__':
    main()





