# Multi-Lingual Text Recognition
pytorch code to train scene text recognition models for indic languages
## MLT19 format to lmdb conversion 
run the following commands if data is in MLT19 format, if data is already in lmdb format skip to Data Setup 
 ``` 
python prepare_data.py --image_dir path/to/SceneImages --image_gt_dir path/to/SceneImage_gt --word_image_dir path/to/store/cropped/text/images --output_path path/to/store/lmdb_gt --lang language to generate gt for
```
```
python create_lmdbdataset.py --gtFile path/to/lmbd_gt --outputPath folder/to/store/lmdb_data
```
## Data Setup
- To get train-test split run
```
python train_test_split_lmdb.py --lmdb_data_path path/to/lmdb_data
```
- Set up data directories as specified in dataDirStruct.txt
## Training
- Refer to Config.py on how to setup a config for an experiment run
- To start training run
```
python train.py --task_config_name <name of the config>
