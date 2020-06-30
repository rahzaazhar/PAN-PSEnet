import os
import numpy as np 
import cv2
import lmdb
import argparse


cnt = 0
def filter_text(lang,text):
  #print(lang,text)
  unicode_range = {'odia':'[^\u0020-\u0040-\u0B00-\u0B7F]','kanada':'[^\u0020-\u0040-\u0C80-\u0CFF]',
  'tamil':'[^\u0020-\u0040-\u0B80-\u0BFF]','malyalam':'[^\u0020-\u0040-\u0D00-\u0D7F]',
  'urdu':'[^\u0020-\u0040-\u0600-\u06FF]','telgu':'[^\u0020-\u0040-\u0C00-\u0C7F]',
  'marathi':'[^\u0020-\u0040-\u0900-\u097F]','sanskrit':'[^\u0020-\u0040-\u0900-\u097F]',
  'hindi':'[^\u0020-\u0040-\u0900-\u097F]','ban':'[^\u0020-\u0040-\u0980-\u09FF]'}
  import re
  t = re.sub(unicode_range[lang],'',text)
  if len(text) == len(t):
    return False
  else:
    return True


def crop(imgpath,bbox):
	img = cv2.imread(imgpath)
	bbox = bbox.reshape(4,2)
	topleft_x = np.min(bbox[:,0])
	topleft_y = np.min(bbox[:,1])
	bot_right_x = np.max(bbox[:,0])
	bot_right_y = np.max(bbox[:,1])
	cropped_img = img[topleft_y:bot_right_y, topleft_x:bot_right_x]
	return cropped_img
	

def create_lists(args):
	global cnt
	imgPathList = []
	labelList = []
	if not os.path.exists(args.word_image_dir):
		os.mkdir(args.word_image_dir)

	gt_filelist = os.listdir(args.image_gt_dir)
	gt_filelist.sort()
	#gt_filelist = gt_filelist[0:9801]

	for gt_file in gt_filelist:
		gt_filename = gt_file.split('.')[0]
		f = open(args.image_gt_dir+gt_file,'r')
		lines = f.readlines()
		for line in lines:
			elements = line.split(',')
			elements[-1] = elements[-1].strip()
			elements[-2] = elements[-2].lower()
			#print(elements[-1])
			#if not filter_text(args.lang,elements[-1]):
			if not (elements[-2]=='Symbol' or elements[-1]=="###" or elements[-1]=='') and args.lang in elements[-2]:
				bbox = [int(ele) for ele in elements[0:8]]
				bbox = np.array(bbox)
				label = elements[-1].strip()
				if cnt<10:
					print(label)
				imgpath = args.image_dir+gt_filename+'.jpg'
				try:
					cropped_img = crop(imgpath,bbox)

				except:
					print(".png image ignore")
					continue
				#print(np.shape(cropped_img))
				if not (0 in np.shape(cropped_img)):
					word_image_path = args.word_image_dir+"img"+str(cnt)+'.jpg'
					print(word_image_path)
					cv2.imwrite(word_image_path,cropped_img)
					imgPathList.append(word_image_path)
					labelList.append(label)
					cnt = cnt+1
					print('processed number:',cnt)
	return imgPathList, labelList

def generate_gt(outputPath,imgPathList,labelList,lang):
	path = outputPath+lang+'lmdb_data_gt.txt'
	with open(path,'w') as f:
		for image,label in zip(imgPathList,labelList):
			line = image+'\t'+label+'\n'
			f.write(line)	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_dir',help='path to Scene images')
	parser.add_argument('--image_gt_dir',help='path to Scene gt')
	parser.add_argument('--word_image_dir',help='path to store cropped word images')
	parser.add_argument('--output_path',help='path to save gt.txt')
	parser.add_argument('--lang',help='language to generate gt for')
	args = parser.parse_args()
	
	imgPathList, labelList = create_lists(args)
	#print(len(imgPathList),len(labelList))

	imgPathList_filtered, labelList_filtered = [], []

	for img, label in zip(imgPathList,labelList):
		if not filter_text(args.lang,label):
			imgPathList_filtered.append(img)
			labelList_filtered.append(label)


	print(len(imgPathList_filtered),len(labelList_filtered))
	generate_gt(args.output_path,imgPathList_filtered,labelList_filtered,args.lang)
    





