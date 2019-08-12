import cv2
import pytesseract
from dataset import IC19TestLoader
from util.io_ import read_lines
from util.io_ import write_lines
import pytesseract as pyt
import os
from os import listdir
import argparse
import numpy as np 

ic19_root_dir = '/home/azhar/summerResearch/data/IC19MLT_root/' 
ic19_train_data_dir = ic19_root_dir + 'train_IC19MLT/'


def crop(imgpath,bbox):
	img = cv2.imread(imgpath)
	bbox = bbox.reshape(4,2)
	topleft_x = np.min(bbox[:,0])
	topleft_y = np.min(bbox[:,1])
	bot_right_x = np.max(bbox[:,0])
	bot_right_y = np.max(bbox[:,1])
	cropped_img = img[topleft_y:bot_right_y, topleft_x:bot_right_x]
	return cropped_img

def recognise(args):
	files = listdir(args.bboxpath)
	for file in files:
		filename = args.bboxpath+file
		#print(filename)
		f = open(filename,'r')
		lines = f.readlines()
		values = []
		for line in lines:
			bbox = line.split(', ')
			bbox = [int(ele) for ele in bbox]
			bbox = np.array(bbox)
			filename = file.split('res_')[-1].split('.')[0]+'.jpg'
			filename = ic19_train_data_dir+filename
			print(filename)
			cropped_img = crop(filename,bbox)
			#print(cropped_img)
			cv2.imshow('image',cropped_img)
			cv2.waitKey(500)
			try:
				text = pyt.image_to_string(cropped_img,lang='hin')
				print(text)
			except ValueError:
				break
			bbox = list(bbox)
			value = "%d,%d,%d,%d,%d,%d,%d,%d"%tuple(bbox)
			value = value+","+text+"\n"
			values.append(value)
		oFilename = args.outpath+file
		write_lines(oFilename,values)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='I/O paths')
	parser.add_argument('--outpath',nargs='?',type=str,default=None,
						help='output path to save results')
	parser.add_argument('--bboxpath',nargs='?',type=str,default=None,
						help='provide path to detected bboxes')
	args = parser.parse_args()
	recognise(args)






