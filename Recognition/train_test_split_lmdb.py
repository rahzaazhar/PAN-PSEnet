import os
import lmdb
import math
import argparse

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def split_if_dir(datadir):
    dirs = os.listdir(datadir)
    for dir_ in dirs:
        lmdb_path = os.path.join(datadir,dir_)
        train_val_split(lmdb_path)

def train_val_split(lmdb_path):
    #input: lmdb_folder or directory containing lmdb_folders
    #output: train_lmdb_folder, val_lmbd_folder or train_lmdb_folders,val_lmbd_folders
    env = lmdb.open(lmdb_path, map_size=1099511627776)
    env_train = lmdb.open(lmdb_path+'_train', map_size=1099511627776)
    env_test = lmdb.open(lmdb_path+'_val', map_size=1099511627776)
    cache_train = {}
    cache_test = {}
    cnt_train = 1
    cnt_test = 1
    with env.begin(write=False) as txn:
        cnt = txn.get('num-samples'.encode())
        cnt = int(cnt.decode())
        i = 1
        thres = math.ceil(0.8*cnt)
        print(thres)
        while i<cnt:
            label_key = 'label-%09d'.encode() % i
            img_key = 'image-%09d'.encode() % i
            imgbuf = txn.get(img_key)
            label = txn.get(label_key)
            if i<=thres:
                imageKey = 'image-%09d'.encode() % cnt_train
                labelKey = 'label-%09d'.encode() % cnt_train
                cache_train[imageKey] = imgbuf
                cache_train[labelKey] = label
                cnt_train+=1
                if cnt_train%500==0:
                    writeCache(env_train, cache_train)
                    print('Written(train) %d / %d' % (cnt_train, thres))
                    cache_train = {}
            else:
                imageKey = 'image-%09d'.encode() % cnt_test
                labelKey = 'label-%09d'.encode() % cnt_test
                cache_test[imageKey] = imgbuf
                cache_test[labelKey] = label
                cnt_test+=1
                if cnt_test%500==0:
                    writeCache(env_test, cache_test)
                    print('Written(test) %d / %d' % (cnt_test, cnt-thres))
                    cache_test = {}
            i=i+1
    cache_train['num-samples'.encode()] = str(cnt_train).encode()
    cache_test['num-samples'.encode()] = str(cnt_test).encode()
    writeCache(env_train,cache_train`)
    writeCache(env_test,cache_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_data_path', required=True, help='path to generated lmdb data')
    parser.add_argument('--dir',action='store_true',dest='dir',default=False,help='true if ')
    args = parser.parse_args()
    if args.dir:
        split_if_dir(args.lmdb_data_path)
    else:
        train_val_split(args.lmdb_data_path)
