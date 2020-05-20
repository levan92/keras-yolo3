import cv2
import time
import argparse
from pathlib import Path

from yolo import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('video_path', help='path to video')
parser.add_argument('--thresh', help='OD confidence threshold', default=0.4, type=float)
args = parser.parse_args()

assert args.thresh > 0.0

yolo = YOLO(bgr=True, 
            batch_size=1, 
            model_path='model_data/yolov3.h5',
            anchors_path='model_data/yolo_anchors.txt',
            classes_path='model_data/coco_classes.txt',
            score=args.thresh,
            model_image_size=(608, 608)
            )

vp = Path(args.video_path)
assert vp.is_file(),'{} not a file'.format(vp)
cap = cv2.VideoCapture(str(vp))
assert cap.isOpened(),'Cannot open video file!'

cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)

while True:
    # Decode
    ret, frame = cap.read()
    if not ret:
        break
    # Inference
    tic = time.perf_counter()
    dets = yolo.detect_get_box_in([frame], box_format='ltrb')
    toc = time.perf_counter()
    print('infer duration: {:0.3f}s'.format(toc-tic))
    dets = dets[0]

    # Drawing
    show_frame = frame.copy()
    for det in dets:
        ltrb, conf, clsname = det
        l,t,r,b = ltrb
        cv2.rectangle(show_frame, (int(l),int(t)),(int(r),int(b)), (255,255,0))
        cv2.putText(show_frame, '{}:{:0.2f}'.format(clsname, conf), (l,b), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255,255,0), lineType=2)

    cv2.imshow('yolo', show_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

 