import cv2
import time
import argparse
from pathlib import Path

from yolo import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='path to image or folder of images')
parser.add_argument('--bs', help='OD Batch Size', default=1, type=int)
parser.add_argument('--thresh', help='OD confidence threshold', default=0.4, type=float)
args = parser.parse_args()

assert args.bs >= 1
assert args.thresh > 0.0

yolo = YOLO(bgr=True, 
            batch_size=args.bs, 
            model_path='model_data/yolov3.h5',
            anchors_path='model_data/yolo_anchors.txt',
            classes_path='model_data/coco_classes.txt',
            score=args.thresh,
            model_image_size=(608, 608)
            )

ip = Path(args.image_path)
images = []
image_paths = []
first_size = None
if ip.is_file():
    images.append(cv2.imread(str(ip))) 
    image_paths.append(ip)
elif ip.is_dir():
    # Note that all images of a batch needs to be same size for the post-processing to make sense. 
    for impath in ip.rglob('*'):
        if impath.suffix in ['.png','.jpg','.jpeg']:
            img = cv2.imread(str(impath))
            if first_size is None:
                first_size = img.shape[:2]
            else:
                img = cv2.resize(img, first_size[::-1])
            images.append(img)
            image_paths.append(impath)
else:
    raise Exception('Path given does not exist')

# Inference
tic = time.perf_counter()
all_dets = yolo.detect_get_box_in(images, box_format='ltrb')
toc = time.perf_counter()
print('infer duration: {:0.3f}s'.format(toc-tic))

# Drawing
for img, imgpath, dets in zip(images, image_paths, all_dets):
    show_img = img.copy()
    for det in dets:
        ltrb, conf, clsname = det
        l,t,r,b = ltrb
        cv2.rectangle(show_img, (int(l),int(t)),(int(r),int(b)), (255,255,0))
        cv2.putText(show_img, '{}:{:0.2f}'.format(clsname, conf), (l,b), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255,255,0), lineType=2)

    newpath = imgpath.parent / '{}_out.jpg'.format(imgpath.stem)
    cv2.imwrite(str(newpath), show_img)
 