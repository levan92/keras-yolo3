# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import os
import cv2
import colorsys
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
CWD = os.getcwd()
if CWD == THIS_DIR:
    from yolo3.model import yolo_eval, yolo_eval_batch, yolo_body, tiny_yolo_body
    from yolo3.utils import letterbox_image
else:
    from .yolo3.model import yolo_eval, yolo_eval_batch, yolo_body, tiny_yolo_body
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
        # "input_image_size" : (1080, 1920), # Height, Width
        "gpu_num" : 1,
        "batch_size" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, bgr, pillow=False, gpu_usage = 0.5, old=False, **kwargs):
        '''
        Params
        ------
        - bgr : Boolean, signifying if the inputs is bgr or rgb (if you're using cv2.imread it's probably in BGR) 
        - pillow : Boolean, flag to give inputs in pillow format instead of ndarray-like, this will override bgr flag to False
        - batch_size : int, inference batch size (default = 1)
        '''
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.bgr = bgr
        self.pillow = pillow
        if self.pillow:
            self.bgr = False
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth=True
        # config.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.set_session(sess)
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate(old=old)
        # self.boxes, self.scores, self.classes = self.generate()
        warmup_image = np.zeros((10,10,3), dtype='uint8')
        # warmup_image = Image.fromarray(np.zeros((10,10,3), dtype='uint8'))
        # self._detect(warmup_image)
        # self._detect([warmup_image, warmup_image])
        print('Warming up...')
        self._detect_batch([warmup_image] * self.batch_size)
        print('YOLO warmed up!')
        # print('Input image size initialised as {}x{} (WxH)! Please give the appropriate argument inputs if this is wrong.'.format(self.input_image_size[1], self.input_image_size[0]))

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

    def generate(self, old = False):
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

        if old:
            print('using old yolo eval')
            boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                    len(self.class_names), self.input_image_shape,
                    score_threshold=self.score, iou_threshold=self.iou)
        else:
            boxes, scores, classes = yolo_eval_batch(self.yolo_model.output, self.anchors,
                    len(self.class_names), self.input_image_shape, self.batch_size,
                    score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes
        # return boxes, scores, classes

    def _refresh(self, batch_size):
        self.batch_size = batch_size
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

        # boxes, scores, classes = yolo_eval_batch(self.yolo_model.output, self.anchors,
        #         len(self.class_names), self.input_image_shape, batch_size=2,
        #         score_threshold=self.score, iou_threshold=self.iou)
        # boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
        #         len(self.class_names), self.input_image_shape,
        #         score_threshold=self.score, iou_threshold=self.iou)

        self.boxes, self.scores, self.classes = yolo_eval_batch(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape, self.batch_size,
                score_threshold=self.score, iou_threshold=self.iou)

    def regenerate(self, batch_size):
        if batch_size == self.batch_size:
            return
        self._refresh(batch_size)
        warmup_image = np.zeros((10,10,3), dtype='uint8')
        print('Warming up...')
        self._detect_batch([warmup_image] * self.batch_size)
        print('YOLO warmed up!')

    def _preprocess(self, image, expand=True):
        '''
        Params
        ------
        image : ndarray-like or PIL image (in that case, self.pillow better be True)
        expand : Boolean, usually True for single image cases

        Returns
        -------
        ndarray-like
        '''

        if not self.pillow:
            if self.bgr: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray( image )
        
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
        if expand:
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        return image_data

    def _detect(self, image):
        image_data = self._preprocess(image)
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]], # height, width
                K.learning_phase(): 0
            })
        return out_boxes, out_scores, out_classes

    def _preprocess_batch(self, images):
        # images_data = np.array( [self._preprocess(image) for image in images] )
        images_data = np.zeros((len(images),*self.model_image_size,images[0].shape[-1]))
        for i, image in enumerate(images):
            if image is not None:
                images_data[i] = self._preprocess( image, expand=False )
        return images_data

    def _detect_batch(self, images):
        '''
        detect function 

        Params
        ------
        images : list of ndarrays

        '''
        if len( images ) <= 0:
            return None
        # assert all([images[0].shape == img.shape for img in images[1:]]),'Network does not acccept images of different sizes. please speak to evan.'

        assert len(images) <= self.batch_size,'Length of image batch given ({}) is bigger than what network was initialised as ({}).'.format(len(images), self.batch_size)
        # assert len(images) == self.batch_size,'Length of image batch given ({}) different from what network was initialised as ({}).'.format(len(images), self.batch_size)
        if len(images) < self.batch_size:
            images.extend([None]*int(self.batch_size - len(images)))
        assert len(images) == self.batch_size
        images_data = self._preprocess_batch(images)
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: images_data,
                self.input_image_shape: [images[0].shape[0], images[0].shape[1]], # height, width
                # self.input_image_shape: [self.input_image_size[0], self.input_image_size[1]], # height, width
                K.learning_phase(): 0
            })
        return out_boxes, out_scores, out_classes
        # return out_boxes, out_scores, out_classes

    def detect_get_box_in(self, images, box_format='ltrb', classes=None, buffer_ratio=0.):
        '''
        Params
        ------
        - images : ndarray-like or list of ndarray-like
        - box_format : string of characters representing format order, where l = left, t = top, r = right, b = bottom, w = width and h = height
        - classes : list of string, classes to focus on
        - buffer : float, proportion of buffer around the width and height of the bounding box

        Returns
        -------
        if one ndarray given, this returns a list (boxes in one image) of tuple (box_infos, score, predicted_class),
        
        else if a list of ndarray given, this return a list (batch) containing the former as the elements,

        where,
            - box_infos : list of floats in the given box format
            - score : float, confidence level of prediction
            - predicted_class : string

        '''
        
        no_batch = False
        if isinstance(images, list):
            if len(images) <= 0 : 
                return None
            else:
                assert isinstance(images[0], np.ndarray)
                if all([images[0].shape == img.shape for img in images[1:]]):
                    print('WARNING from yolo module: Input images in batch are of diff sizes, the input size will take the first image in the batch, you will have to scale the output bounding boxes of those input image whose sizes differ from the first image yourself.')
                # assert all([images[0].shape == img.shape for img in images[1:]]),'Network does not acccept images of different sizes. please speak to eugene.'
        elif isinstance(images, np.ndarray):
            images = [ images ]
            no_batch = True
        im_height, im_width = images[0].shape[:2]

        # import time
        # tic = time.time()
        all_out_boxes = []
        all_out_scores = []
        all_out_classes = []
        for i in range( int(np.ceil(len(images)/self.batch_size)) ):
            from_ = i*self.batch_size
            to_ = min(len(images),i*self.batch_size+self.batch_size)
            n = to_ - from_ 
            # print('Inferencing {} images'.format(n))
            out_boxes, out_scores, out_classes = self._detect_batch(images[from_:to_])
            all_out_boxes.extend(out_boxes[:n])
            all_out_scores.extend(out_scores[:n])
            all_out_classes.extend(out_classes[:n])
        # tic2 = time.time()
        all_dets = []
        for out_boxes, out_scores, out_classes in zip(all_out_boxes, all_out_scores, all_out_classes):
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
                width_buffer = width * buffer_ratio
                height_buffer = height * buffer_ratio
                
                top = max( 0.0, top-0.5*height_buffer )
                left = max( 0.0, left-0.5*width_buffer )
                bottom = min( im_height - 1.0, bottom + 0.5*height_buffer )
                right = min( im_width - 1.0, right + 0.5*width_buffer )

                box_infos = []
                for c in box_format:
                    if c == 't':
                        box_infos.append( int(round(top)) ) 
                    elif c == 'l':
                        box_infos.append( int(round(left)) )
                    elif c == 'b':
                        box_infos.append( int(round(bottom)) )
                    elif c == 'r':
                        box_infos.append( int(round(right)) )
                    elif c == 'w':
                        box_infos.append( int(round(width+width_buffer)) )
                    elif c == 'h':
                        box_infos.append( int(round(height+height_buffer)) )
                    else:
                        assert False,'box_format given in detect unrecognised!'
                assert len(box_infos) > 0 ,'box infos is blank'

                dets.append( (box_infos, score, predicted_class) )
                # dets.append((top, left, bottom, right) (predicted_class, score, ) )
            all_dets.append(dets)
        # tic3 = time.time()
        # print('Batch Forward pass: {}s'.format(tic2 - tic))
        # print('Post proc: {}s'.format(tic3 - tic2))
        if no_batch:
            return all_dets[0]
        else:
            return all_dets

    def detect_ltwh(self, np_image, classes=None, buffer=0.):
        raise Exception('This method has been deprecated, please use detect_get_box_in for a more general method.')
        '''
        detect method

        Params
        ------
        np_image : ndarray

        Returns
        ------
        list of triples ([left, top, width, height], score, predicted_class)

        '''
        # image = Image.fromarray(np_image)
        # if self.model_image_size != (None, None):
        #     assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
        #     assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
        #     boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        # else:
        #     new_image_size = (image.width - (image.width % 32),
        #                       image.height - (image.height % 32))
        #     boxed_image = letterbox_image(image, new_image_size)
        # image_data = np.array(boxed_image, dtype='float32')

        # image_data /= 255.
        # image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        # image_data = self.preprocess(image)

        # out_boxes, out_scores, out_classes = self.sess.run(
        #     [self.boxes, self.scores, self.classes],
        #     feed_dict={
        #         self.yolo_model.input: image_data,
        #         self.input_image_shape: [image.size[1], image.size[0]],
        #         K.learning_phase(): 0
        #     })

        # image_data = self.preprocess(image)
        # out_boxes, out_scores, out_classes = self.sess.run(
        #     [self.boxes, self.scores, self.classes],
        #     feed_dict={
        #         self.yolo_model.input: image_data,
        #         self.input_image_shape: [image_data.shape[0], image_data.shape[1]], # height, width
        #         K.learning_phase(): 0
        #     })

        # dets = []

        # for i, c in reversed(list(enumerate(out_classes))):
        #     predicted_class = self.class_names[c]
        #     if classes is not None and predicted_class not in classes:
        #         continue

        #     score = out_scores[i]
        #     box = out_boxes[i]
        #     top, left, bottom, right = box
        #     width = right - left + 1
        #     height = bottom - top + 1
        #     width_buf = (width) * buffer
        #     height_buf = (height) * buffer
        #     top = max(0, np.floor(top + 0.5 - height_buf).astype('int32'))
        #     left = max(0, np.floor(left + 0.5 - width_buf).astype('int32'))
        #     bottom = min(image.size[1], np.floor(bottom + 0.5 + height_buf).astype('int32'))
        #     right = min(image.size[0], np.floor(right + 0.5 + width_buf).astype('int32'))

        #     dets.append( ([left, top, width, height], score, predicted_class) )

        # return dets

    def detect_path(self, path, box_format='ltrb', classes=None):
        img = cv2.imread(path)
        assert self.pillow == False,'Please initialise this object with pillow = False'
        assert self.bgr == True,'Please initialise this object with bgr = True'
        # print('WARNING: pillow set to False and bgr set to True, do not use this yolo object for anything else.')
        dets = self.detect_get_box_in(img, box_format='ltrb', classes=classes)
        return dets

    # def detect_persons(self, image, classes=None, buf=0.):
    #     return self.detect( image, classes=['person'], buffer=buf )

    def get_detections_dict(self, frames, classes=None, buffer_ratio=0.0):
        '''
        Params: frames, list of ndarray-like
        Returns: detections, list of dict, whose key: label, confidence, t, l, w, h
        '''
        if frames is None or len(frames) == 0:
            return None
        all_dets = self.detect_get_box_in( frames, box_format='tlbrwh', classes=classes, buffer_ratio=buffer_ratio )
        
        all_detections = []
        for dets in all_dets:
            detections = []
            for tlbrwh,confidence,label in dets:
                top, left, bot, right, width, height = tlbrwh
                # left = tlbr[1]
                # bot = tlbr[2]
                # right = tlbr[3]
                # width = right - left
                # height = bot - top
                detections.append( {'label':label,'confidence':confidence,'t':top,'l':left,'b':bot,'r':right,'w':width,'h':height} ) 
            all_detections.append(detections)
        return all_detections

    def get_triple_detections(self, frame, classes=None):
        raise Exception('this method has been deprecated, please use detect_get_box_in for a more general method.')
        # '''
        # Params
        # ------
        # frame : np array
        
        # Returns
        # ------
        # list
        #     List of triples ( [left,top,w,h] , confidence, detection_class)

        # '''
        # if frame is None:
        #     return None
        # image = Image.fromarray( frame )
        # dets = self.detect( image, classes=classes )
        # detections = []
        # for label, confidence, tlbr in dets:
        #     top = tlbr[0]
        #     left = tlbr[1]
        #     bot = tlbr[2]
        #     right = tlbr[3]
        #     width = right - left
        #     height = bot - top
        #     detections.append( ([left, top, width, height], confidence, label) ) 
        # return detections

    # for reid PERSON ONLY
    def get_detections_batch(self, frames):
        # TODO: BATCH INFER THIS SHIT
        all_detections = []
        for frame in frames:
            if frame is None:
                all_detections.append([])
                continue
            curr_detections = self.get_detections_dict(frame, classes=['person'])
            # image = Image.fromarray( frame )
            # dets = self.detect_get_box_in( image,  classes=['person'] )
            # curr_detections = []
            # for label, confidence, tlbr in dets:
            #     top = tlbr[0]
            #     left = tlbr[1]
            #     bot = tlbr[2]
            #     right = tlbr[3]
            #     width = right - left
            #     height = bot - top
            #     tlwh = {'t':top, 'l':left, 'w':width, 'h':height}
            #     curr_detections.append( {'label':label, 'confidence':confidence, 'tlwh':tlwh} )
            all_detections.append(curr_detections)
        return all_detections

    def crop_largest_person(self, image, buf=0.1):
        raise Exception('this method has been deprecated, please use detect_get_box_in for a more general method.')
        # dets = self.detect( image, classes=['person'], buffer=buf )
        # # get the largest detection
        # largest_det = None
        # for _,_,tlbr in dets:
        #     if largest_det is None:
        #         largest_det = tlbr
        #     else:
        #         detarea = (tlbr[3]-tlbr[1]) * (tlbr[2]-tlbr[0])
        #         ldarea = (largest_det[3]-largest_det[1]) * (largest_det[2]-largest_det[0])
        #         if detarea > ldarea:
        #             largest_det = tlbr

        # if largest_det is not None:
        #     # crop image
        #     min_x = largest_det[1]
        #     min_y = largest_det[0]
        #     max_x = largest_det[3]
        #     max_y = largest_det[2]
        #     return image.crop( (min_x, min_y, max_x, max_y) )
        # return image

    def get_largest_person(self, np_image, buf=0.1):
        raise Exception('this method has been deprecated, please use detect_get_box_in for a more general method.')
        # # image = Image.fromarray(np_image)
        # dets = self.detect_ltwh( np_image, classes=['person'], buffer=buf )
        # # get the largest detection
        # largest_det = None
        # for det in dets:
        #     if largest_det is None:
        #         largest_det = det
        #     else:
        #         detarea = det[0][2] * det[0][3]
        #         ldarea = largest_det[0][2] * largest_det[0][3]
        #         if detarea > ldarea:
        #             largest_det = det
        # return largest_det

    def get_largest_person_and_bb(self, np_image, buf=0.1):
        raise Exception('this method has been deprecated, please use detect_get_box_in for a more general method.')
        # image = Image.fromarray(np_image)
        # dets = self.detect( image, classes=['person'], buffer=buf )
        # # get the largest detection
        # largest_det = None
        # for _,_,tlbr in dets:
        #     if largest_det is None:
        #         largest_det = tlbr
        #     else:
        #         detarea = (tlbr[3]-tlbr[1]) * (tlbr[2]-tlbr[0])
        #         ldarea = (largest_det[3]-largest_det[1]) * (largest_det[2]-largest_det[0])
        #         if detarea > ldarea:
        #             largest_det = tlbr

        # if largest_det is not None:
        #     # crop image
        #     min_x = largest_det[1]
        #     min_y = largest_det[0]
        #     max_x = largest_det[3]
        #     max_y = largest_det[2]
        #     return np.array(image.crop( (min_x, min_y, max_x, max_y) )), largest_det
        # return None, None

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
    import time
    # yolo = YOLO(bgr=True, batch_size=1)
    
    # img = cv2.imread('/home/dh/Pictures/frisbee.jpg')
    # img2 = cv2.imread('/home/dh/Pictures/dog_two.jpg')
    # img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    # img3 = cv2.imread('/home/dh/Pictures/puppy-dog.jpg')
    # img3 = cv2.resize(img3, (img.shape[1], img.shape[0]))

    # img_batch = [img]
    # # img_batch = [img, img2]
    # # img_batch = [img, img2, img3]

    # all_dets = yolo.detect_get_box_in(img_batch, box_format='ltrb')
    # # boxes, scores, classes = yolo._detect_batch(img_batch)
    # for dets, im in zip(all_dets, img_batch):
    #     im_show = im.copy()
    #     for det in dets:
    #         # print(det)
    #         ltrb, conf, clsname = det
    #         l,t,r,b = ltrb
    #         cv2.rectangle(im_show, (int(l),int(t)),(int(r),int(b)), (255,255,0))
    #         print('{}:{}'.format(clsname, conf))
    #     cv2.imshow('',im_show)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print('REDO with diff batch size')

    # img_batch = [img, img2, img3]
    # yolo.regenerate(batch_size=len(img_batch))
    # all_dets = yolo.detect_get_box_in(img_batch, box_format='ltrb')
    # for dets, im in zip(all_dets, img_batch):
    #     im_show = im.copy()
    #     for det in dets:
    #         # print(det)
    #         ltrb, conf, clsname = det
    #         l,t,r,b = ltrb
    #         cv2.rectangle(im_show, (int(l),int(t)),(int(r),int(b)), (255,255,0))
    #         print('{}:{}'.format(clsname, conf))
    #     cv2.imshow('',im_show)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    numstreams = 20
    bs = 8

    vp = '/media/dh/HDD/reid/street_looped.mp4'
    cap = cv2.VideoCapture(vp)
    caps = []
    for _ in range(numstreams):
        caps.append(cv2.VideoCapture(vp))
    yolo = YOLO(bgr=True, batch_size=bs)

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            # frames = [frame] * numstreams

        tic = time.time()
        all_dets = yolo.detect_get_box_in(frames, box_format='ltrb')
        toc = time.time()
        print('infer time:', toc-tic)
        # for dets, im in zip(all_dets, frames):
        im_show = frame.copy()
        for det in all_dets[0]:
            # print(det)
            ltrb, conf, clsname = det
            l,t,r,b = ltrb
            cv2.rectangle(im_show, (int(l),int(t)),(int(r),int(b)), (255,255,0))
            # print('{}:{}'.format(clsname, conf))
        cv2.imshow('',im_show)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
