# Scene Text-Spotting based on PSEnet+CRNN
Pytorch implementation of an end to end Text-Spotter with a PSEnet text detector and CRNN text recognizer.

## Requirements
- Python 3.6.5
- Pytorch 1.0.1
- pyclipper
- Polygon 3.0.8
- OpenCV 3.4.1

### Testing
- Download the trained CRNN and PSEnet models from the links provided.
- Copy paths of the models and paste them in params.py
- run end-end.py
```
python end-end.py --img [path to image]
```
link to trained models [drive](https://drive.google.com/open?id=1Bza5tAACtqtLNLxg7ws6rzRrSeW1-Jp6) 










