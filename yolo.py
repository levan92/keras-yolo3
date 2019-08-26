# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf

if __name__ == '__main__':
    from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
    from yolo3.utils import letterbox_image
else:
    from .yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
    from .yolo3.utils import letterbox_image

KERAS_YOLO_DIR = os.path.dirname(os.path.abspath(__file__))

class YOLO(object):
    _defaults = {
        # "model_path": os.path.join(KERAS_YOLO_DIR, 'model_data/yolov3-tiny.h5'),
        "model_path": os.path.join(KERAS_YOLO_DIR, 'model_data/yolov3.h5'),
        # "model_path": os.path.join(KERAS_YOLO_DIR, 'model_data/pp_reanchored_best_train.h5'),
        # "model_path": os.path.join(KERAS_YOLO_DIR, 'model_data/pp_reanchored_best_val.h5'),
        # "anchors_path": os.path.join(KERAS_YOLO_DIR, 'model_data/tiny_yolo_anchors.txt'),
        # "anchors_path": os.path.join(KERAS_YOLO_DIR, 'model_data/PP_ALL_anchors.txt'),
        "anchors_path": os.path.join(KERAS_YOLO_DIR, 'model_data/yolo_anchors.txt'),
        "classes_path": os.path.join(KERAS_YOLO_DIR, 'model_data/coco_classes.txt'),
        # "classes_path": os.path.join(KERAS_YOLO_DIR, 'model_data/PP_classes.txt'),
        "score" : 0.5,
        "iou" : 0.45,
        "model_image_size" : (608, 608),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, gpu_usage = 0.5, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth=True
        # config.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.set_session(sess)
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        warmup_image = Image.fromarray(np.zeros((10,10,3), dtype='uint8'))
        self.detect(warmup_image)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def detect(self, image, classes=None, buffer=0.):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        dets = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if classes is not None and predicted_class not in classes:
                continue

            score = out_scores[i]
            box = out_boxes[i]
            top, left, bottom, right = box
            width_buf = (right - left) * buffer
            height_buf = (bottom - top) * buffer
            top = max(0, np.floor(top + 0.5 - height_buf).astype('int32'))
            left = max(0, np.floor(left + 0.5 - width_buf).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5 + height_buf).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5 + width_buf).astype('int32'))

            dets.append( (predicted_class, score, (top, left, bottom, right)) )

        return dets

    def detect_ltwh(self, np_image, classes=None, buffer=0.):
        '''
        detect method

        Params
        ------
        np_image : ndarray

        Returns
        ------
        list of triples ([left, top, width, height], score, predicted_class)

        '''
        image = Image.fromarray(np_image)
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        dets = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if classes is not None and predicted_class not in classes:
                continue

            score = out_scores[i]
            box = out_boxes[i]
            top, left, bottom, right = box
            width = right - left + 1
            height = bottom - top + 1
            width_buf = (width) * buffer
            height_buf = (height) * buffer
            top = max(0, np.floor(top + 0.5 - height_buf).astype('int32'))
            left = max(0, np.floor(left + 0.5 - width_buf).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5 + height_buf).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5 + width_buf).astype('int32'))

            dets.append( ([left, top, width, height], score, predicted_class) )

        return dets


    def detect_path(self, path, classes=None):
        img = cv2.imread(path)
        dets = self.detect(img, classes=classes)
        return dets

    # def detect_persons(self, image, classes=None, buf=0.):
    #     return self.detect( image, classes=['person'], buffer=buf )

    def get_detections(self, frame, classes=None):
        '''
        Params: frame, np array
        Returns: detections, list of dict, whose key: label, confidence, t, l, w, h
        '''
        if frame is None:
            return None
        image = Image.fromarray( frame )
        dets = self.detect( image, classes=classes )
        detections = []
        for label, confidence, tlbr in dets:
            top = tlbr[0]
            left = tlbr[1]
            bot = tlbr[2]
            right = tlbr[3]
            width = right - left
            height = bot - top
            detections.append( {'label':label,'confidence':confidence,'t':top,'l':left,'b':bot,'r':right,'w':width,'h':height} ) 
        return detections

    def get_triple_detections(self, frame, classes=None):
        '''
        Params
        ------
        frame : np array
        
        Returns
        ------
        list
            List of triples ( [left,top,w,h] , confidence, detection_class)

        '''
        if frame is None:
            return None
        image = Image.fromarray( frame )
        dets = self.detect( image, classes=classes )
        detections = []
        for label, confidence, tlbr in dets:
            top = tlbr[0]
            left = tlbr[1]
            bot = tlbr[2]
            right = tlbr[3]
            width = right - left
            height = bot - top
            detections.append( ([left, top, width, height], confidence, label) ) 
        return detections


    # for reid PERSON ONLY
    def get_detections_batch(self, frames):
        all_detections = []
        for frame in frames:
            if frame is None:
                all_detections.append([])
                continue
            image = Image.fromarray( frame )
            dets = self.detect( image, classes=['person'] )
            curr_detections = []
            for label, confidence, tlbr in dets:
                top = tlbr[0]
                left = tlbr[1]
                bot = tlbr[2]
                right = tlbr[3]
                width = right - left
                height = bot - top
                tlwh = {'t':top, 'l':left, 'w':width, 'h':height}
                curr_detections.append( {'label':label, 'confidence':confidence, 'tlwh':tlwh} )
            all_detections.append(curr_detections)
        return all_detections

    def crop_largest_person(self, image, buf=0.1):
        dets = self.detect( image, classes=['person'], buffer=buf )
        # get the largest detection
        largest_det = None
        for _,_,tlbr in dets:
            if largest_det is None:
                largest_det = tlbr
            else:
                detarea = (tlbr[3]-tlbr[1]) * (tlbr[2]-tlbr[0])
                ldarea = (largest_det[3]-largest_det[1]) * (largest_det[2]-largest_det[0])
                if detarea > ldarea:
                    largest_det = tlbr

        if largest_det is not None:
            # crop image
            min_x = largest_det[1]
            min_y = largest_det[0]
            max_x = largest_det[3]
            max_y = largest_det[2]
            return image.crop( (min_x, min_y, max_x, max_y) )
        return image

    def get_largest_person(self, np_image, buf=0.1):
        # image = Image.fromarray(np_image)
        dets = self.detect_ltwh( np_image, classes=['person'], buffer=buf )
        # get the largest detection
        largest_det = None
        for det in dets:
            if largest_det is None:
                largest_det = det
            else:
                detarea = det[0][2] * det[0][3]
                ldarea = largest_det[0][2] * largest_det[0][3]
                if detarea > ldarea:
                    largest_det = det
        return largest_det

    def get_largest_person_and_bb(self, np_image, buf=0.1):
        image = Image.fromarray(np_image)
        dets = self.detect( image, classes=['person'], buffer=buf )
        # get the largest detection
        largest_det = None
        for _,_,tlbr in dets:
            if largest_det is None:
                largest_det = tlbr
            else:
                detarea = (tlbr[3]-tlbr[1]) * (tlbr[2]-tlbr[0])
                ldarea = (largest_det[3]-largest_det[1]) * (largest_det[2]-largest_det[0])
                if detarea > ldarea:
                    largest_det = tlbr

        if largest_det is not None:
            # crop image
            min_x = largest_det[1]
            min_y = largest_det[0]
            max_x = largest_det[3]
            max_y = largest_det[2]
            return np.array(image.crop( (min_x, min_y, max_x, max_y) )), largest_det
        return None, None

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

if __name__ == '__main__':
    import cv2
    yolo = YOLO()
    # img = cv2.imread('/home/levan/Pictures/auba.jpg')
    # image = Image.fromarray( img )

    image = Image.open('/home/levan/Pictures/auba.jpg')
    iw, ih = image.size
    print(iw,ih)

    yolo.detect(image)
    # boxes, scores, classes = yolo.generate()
    # print(boxes)
    # print(scores)
    # print(classes)