import os
import regex
image_gt_dir = '/home/azhar/summerResearch/data/IC19MLT_root/train_IC19MLT_gt/'
alphabets = []
gt_filelist = os.listdir(image_gt_dir)
gt_filelist.sort()
gt_filelist = gt_filelist[9000:10001]
cnt = 0
for gt_file in gt_filelist:
	f = open(image_gt_dir+gt_file,'r')
	lines = f.readlines()
	for line in lines:
		elements = line.split(',')
		if elements[-2]=='Hindi':
			label = elements[-1].strip()
			labellist = regex.findall(r'\X', label)
			print(label)
			print("length of label",len(label))
			print("length of label using regex",len(labellist))
			for i in labellist:
				if i not in alphabets:
					alphabets.append(i)
					cnt = cnt +1
alphabets = list(set(alphabets))
alphabets.sort()

print(''.join(alphabets))
print(cnt)
print("length of alphabets using len()",len(alphabets))


