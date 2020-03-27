import os
import regex
image_gt_dirs = ['/home/azhar/crnn-pytorch/data/marathi/train_gt.txt','/home/azhar/crnn-pytorch/data/marathi/val_gt.txt'] 
#trainMT_gt_hind.txt trainMT_gt_Bangla.txt
alphabets=[]
for image_gt_dir in image_gt_dirs:
	f = open(image_gt_dir,'r')
	lines = f.readlines()
	for line in lines:
		elements = line.split('\t')
		label = elements[-1].strip()
		for i in label:
			if i not in alphabets:
				alphabets.append(i)

alphabets = ''.join(alphabets)
alphabets = list(set(alphabets.lower()))
alphabets.sort()
alphabets = ''.join(alphabets)
alphabets = 'mar,'+alphabets
print(alphabets)
print(len(alphabets))

#f1 = open('characters.txt','a+')
#f1.write(alphabets)

fr = open('characters.txt','r')
line  = fr.readlines()
print(line)

