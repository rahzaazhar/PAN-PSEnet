import os
import numpy as np 
import cv2
import lmdb

'''image_dir = '/home/azhar/summerResearch/data/IC19MLT_root/train_IC19MLT/'
image_gt_dir = '/home/azhar/summerResearch/data/IC19MLT_root/train_IC19MLT_gt/'
word_image_dir = '/home/azhar/crnn-pytorch/data/word_imagesMLT1/'
outputPath = '/home/azhar/crnn-pytorch/data/'''

image_dir='/home/azhar/crnn-pytorch/data/syn/syn_bangla/Scene_images/'
image_gt_dir = '/home/azhar/crnn-pytorch/data/syn/syn_bangla/Scene_gt/'
word_image_dir ='/home/azhar/crnn-pytorch/data/syn/syn_bangla/word_images/'
outputPath = '/home/azhar/crnn-pytorch/data/syn/syn_bangla/'
cnt = 0


def crop(imgpath,bbox):
	img = cv2.imread(imgpath)
	bbox = bbox.reshape(4,2)
	topleft_x = np.min(bbox[:,0])
	topleft_y = np.min(bbox[:,1])
	bot_right_x = np.max(bbox[:,0])
	bot_right_y = np.max(bbox[:,1])
	cropped_img = img[topleft_y:bot_right_y, topleft_x:bot_right_x]
	return cropped_img
	

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
		f = open(image_gt_dir+gt_file,'r')
		lines = f.readlines()
		for line in lines:
			elements = line.split(',')
			elements[-1] = elements[-1].strip()
			#print(elements[-1])
			if not (elements[-2]=='Symbol' or elements[-1]=="###" or elements[-1]==''):
				bbox = [int(ele) for ele in elements[0:8]]
				bbox = np.array(bbox)
				label = elements[-1].strip()
				if cnt<10:
					print(label)
				imgpath = image_dir+gt_filename+'.jpg'
				try:
					cropped_img = crop(imgpath,bbox)

				except:
					print(".png image ignore")
					continue
				#print(np.shape(cropped_img))
				if not (0 in np.shape(cropped_img)):
					word_image_path = word_image_dir+"syn_img"+str(cnt)+'.jpg'
					print(word_image_path)
					cv2.imwrite(word_image_path,cropped_img)
					imgPathList.append(word_image_path)
					labelList.append(label)
					cnt = cnt+1
					print('processed number:',cnt)
	return imgPathList, labelList

def generate_gt(outputPath,imagePathList,labelList,test_train_flag):
	if(test_train_flag==1):
		path = outputPath+'train_gt.txt'
	elif(test_train_flag==0):
		path = outputPath+'val_gt.txt'

	with open(path,'w') as f:
		for image,label in zip(imagePathList,labelList):
			line = image+'\t'+label+'\n'
			f.write(line)


def main():
	imgPathList, labelList = create_lists()
	print(len(imgPathList),len(labelList))
	#createDataset(outputPath,imgPathList,labelList)
	split = round(len(imgPathList)*0.8)
	generate_gt(outputPath,imgPathList[0:split],labelList[0:split],1)
	generate_gt(outputPath,imgPathList[split:len(imgPathList)],labelList[split:len(imgPathList)],0)


if __name__ == '__main__':
    main()





