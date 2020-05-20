# keras-yolo3

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) forked from [qqwweee](https://github.com/qqwweee/keras-yolo3). Main contribution over the previous version is that this is able to do batch inference now. This comes with a caveat that the input image size needs to be pre-defined now. 

---

## Quick Start

### Converted hdf5 file
- Download from qqwwee [yolov3.h5](https://drive.google.com/open?id=1MFCC4Rpuhn5clQKM8NWH9X49Fitf1QCd)

### Do the conversion yourself
```
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```

## Usage
- `yolo.py` is intended for use as an object, for example `from yolo import YOLO`, then instantiate the YOLO class as an object. 

- Take note of the default parameter by looking into `yolo.py`. One important parameter often overlooked is the input image size. 

- There are many methods in the class due to different projects needing it and legacy reasons, but the main method to use is `detect_get_box_in` where you give a list of ndarray-like images and can specify the format you want the BBs back in.  

## Example usage
Take a look at `example_video.py` on how I will use it on a video

## Details
This implementation is special/weird in the sense that while the inference is in Keras, but the preprocessing is done with tensorflow aka computation graphs.  